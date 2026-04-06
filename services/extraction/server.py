"""Persephone Extraction Service — gRPC server.

Implements the ExtractionService proto contract. Routes documents to
the appropriate backend (Docling for PDF, native parser for LaTeX,
plain read for text/markdown).

Usage:
    python -m extraction.server
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import stat
import sys
import tempfile
import time
from pathlib import Path

import grpc
from grpc import aio as grpc_aio

# Ensure generated stubs are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))

from persephone.extraction import extraction_pb2, extraction_pb2_grpc  # noqa: E402

from .config import ExtractionConfig  # noqa: E402
from .docling_backend import DoclingExtractor, ExtractionResult  # noqa: E402
from .latex_backend import LaTeXExtractor  # noqa: E402

logger = logging.getLogger(__name__)

# Extension → SourceType mapping
_EXT_TO_SOURCE: dict[str, int] = {
    ".pdf": extraction_pb2.SOURCE_TYPE_PDF,
    ".tex": extraction_pb2.SOURCE_TYPE_LATEX,
    ".gz": extraction_pb2.SOURCE_TYPE_LATEX,  # arXiv .tar.gz / .gz
    ".py": extraction_pb2.SOURCE_TYPE_CODE,
    ".rs": extraction_pb2.SOURCE_TYPE_CODE,
    ".cu": extraction_pb2.SOURCE_TYPE_CODE,
    ".cpp": extraction_pb2.SOURCE_TYPE_CODE,
    ".c": extraction_pb2.SOURCE_TYPE_CODE,
    ".js": extraction_pb2.SOURCE_TYPE_CODE,
    ".ts": extraction_pb2.SOURCE_TYPE_CODE,
    ".go": extraction_pb2.SOURCE_TYPE_CODE,
    ".java": extraction_pb2.SOURCE_TYPE_CODE,
    ".md": extraction_pb2.SOURCE_TYPE_MARKDOWN,
    ".markdown": extraction_pb2.SOURCE_TYPE_MARKDOWN,
    ".txt": extraction_pb2.SOURCE_TYPE_TEXT,
    ".text": extraction_pb2.SOURCE_TYPE_TEXT,
    ".rst": extraction_pb2.SOURCE_TYPE_TEXT,
}


class ExtractionServicer(extraction_pb2_grpc.ExtractionServiceServicer):
    """gRPC servicer implementing the ExtractionService proto."""

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._docling: DoclingExtractor | None = None
        self._latex = LaTeXExtractor()
        self._last_request_time = time.time()
        self._active_requests = 0

    def _get_docling(self) -> DoclingExtractor:
        """Lazy-load the Docling extractor."""
        if self._docling is None:
            self._docling = DoclingExtractor(
                use_ocr=self._config.use_ocr,
                use_fallback=self._config.use_fallback,
            )
        return self._docling

    def _detect_source_type(self, request: extraction_pb2.ExtractRequest) -> int:
        """Detect source type from request or file extension."""
        if request.source_type != extraction_pb2.SOURCE_TYPE_UNKNOWN:
            return request.source_type

        path = Path(request.file_path)
        name = path.name.lower()

        # Handle .tar.gz specially
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            return extraction_pb2.SOURCE_TYPE_LATEX

        return _EXT_TO_SOURCE.get(path.suffix.lower(), extraction_pb2.SOURCE_TYPE_UNKNOWN)

    async def Extract(self, request, context):
        """Extract structured content from a document."""
        source_type = self._detect_source_type(request)

        try:
            type_name = extraction_pb2.SourceType.Name(source_type)
        except ValueError:
            type_name = f"UNKNOWN({source_type})"

        logger.info(
            "Extract request: path=%s, source_type=%s",
            request.file_path,
            type_name,
        )

        # If content bytes provided, write to temp file
        tmp_path = None
        if request.content:
            suffix = Path(request.file_path).suffix if request.file_path else ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(request.content)
            tmp.close()
            tmp_path = tmp.name
        file_path = tmp_path or request.file_path

        self._active_requests += 1
        try:
            result = await self._route_extraction(
                file_path, source_type, request
            )
        finally:
            self._active_requests -= 1
            self._last_request_time = time.time()
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        if result.error:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(result.error)
            return extraction_pb2.ExtractResponse()

        # Map to proto response
        tables = [
            extraction_pb2.Table(
                content=t.get("content", ""),
                caption=t.get("caption", ""),
                index=t.get("index", i),
            )
            for i, t in enumerate(result.tables)
        ]
        equations = [
            extraction_pb2.Equation(
                latex=eq.get("latex", ""),
                text=eq.get("text", ""),
                index=eq.get("index", i),
                is_inline=eq.get("is_inline", False),
            )
            for i, eq in enumerate(result.equations)
        ]
        images = [
            extraction_pb2.ImageRef(
                path=img.get("path", ""),
                caption=img.get("caption", ""),
                index=img.get("index", i),
            )
            for i, img in enumerate(result.images)
        ]

        return extraction_pb2.ExtractResponse(
            full_text=result.text,
            tables=tables,
            equations=equations,
            images=images,
            metadata=result.metadata,
            source_type=source_type,
        )

    async def _route_extraction(self, file_path, source_type, request):
        """Route to appropriate backend based on source type."""
        loop = asyncio.get_running_loop()
        if source_type == extraction_pb2.SOURCE_TYPE_PDF:
            return await loop.run_in_executor(
                None,
                lambda: self._get_docling().extract(
                    file_path,
                    extract_tables=request.extract_tables,
                    extract_equations=request.extract_equations,
                    extract_images=request.extract_images,
                    use_ocr=request.use_ocr if request.use_ocr else None,
                ),
            )
        elif source_type == extraction_pb2.SOURCE_TYPE_LATEX:
            return await loop.run_in_executor(
                None,
                lambda: self._latex.extract(
                    file_path,
                    extract_tables=request.extract_tables,
                    extract_equations=request.extract_equations,
                ),
            )
        elif source_type in (
            extraction_pb2.SOURCE_TYPE_TEXT,
            extraction_pb2.SOURCE_TYPE_MARKDOWN,
            extraction_pb2.SOURCE_TYPE_CODE,
        ):
            return await loop.run_in_executor(
                None, lambda: self._extract_text(file_path)
            )
        else:
            # Unknown: try docling (it handles many formats)
            return await loop.run_in_executor(
                None,
                lambda: self._get_docling().extract(
                    file_path,
                    extract_tables=request.extract_tables,
                    extract_equations=request.extract_equations,
                    extract_images=request.extract_images,
                ),
            )

    async def Capabilities(self, request, context):
        """Report extractor capabilities."""
        return extraction_pb2.ExtractorInfo(
            supported_extensions=[
                ".pdf", ".tex", ".gz", ".tar.gz", ".tgz",
                ".md", ".markdown", ".txt", ".text", ".rst",
                ".py", ".rs", ".cu", ".cpp", ".c", ".js", ".ts", ".go", ".java",
            ],
            supported_types=[
                extraction_pb2.SOURCE_TYPE_PDF,
                extraction_pb2.SOURCE_TYPE_LATEX,
                extraction_pb2.SOURCE_TYPE_CODE,
                extraction_pb2.SOURCE_TYPE_MARKDOWN,
                extraction_pb2.SOURCE_TYPE_TEXT,
            ],
            features=["tables", "equations", "images", "ocr", "latex", "fallback"],
            gpu_available=self._check_gpu(),
        )

    @property
    def last_request_time(self) -> float:
        return self._last_request_time

    def unload_models(self) -> None:
        """Unload ML models to free GPU memory."""
        if self._docling is not None:
            self._docling.cleanup()
            self._docling = None
            logger.info("Docling extractor unloaded")

    @staticmethod
    def _extract_text(file_path: str) -> ExtractionResult:
        """Read a plain text file."""
        path = Path(file_path)
        if not path.exists():
            return ExtractionResult(error=f"File not found: {path}")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return ExtractionResult(
                text=text,
                metadata={"source": str(path), "format": path.suffix.lstrip(".")},
            )
        except Exception as e:
            return ExtractionResult(error=f"Failed to read file: {e}")

    @staticmethod
    def _check_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


async def idle_monitor(servicer: ExtractionServicer, config: ExtractionConfig) -> None:
    """Background task to unload models after idle timeout."""
    poll_interval = (
        min(60, max(1, int(config.idle_timeout_seconds)))
        if config.idle_timeout_seconds > 0
        else 60
    )
    while True:
        await asyncio.sleep(poll_interval)
        if (
            config.idle_timeout_seconds > 0
            and servicer._active_requests == 0
            and servicer._docling is not None
            and servicer._docling.is_loaded
            and (time.time() - servicer.last_request_time) > config.idle_timeout_seconds
        ):
            logger.info(
                "Extractor idle for %.0fs — unloading models to free GPU memory",
                config.idle_timeout_seconds,
            )
            servicer.unload_models()


async def serve() -> None:
    """Start the extraction gRPC server."""
    config = ExtractionConfig.from_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    servicer = ExtractionServicer(config)

    server = grpc_aio.server()
    extraction_pb2_grpc.add_ExtractionServiceServicer_to_server(servicer, server)

    socket_path = config.socket_path
    socket_dir = Path(socket_path).parent
    socket_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale socket (only if it's actually a socket)
    sock = Path(socket_path)
    if sock.exists():
        if stat.S_ISSOCK(sock.stat().st_mode):
            sock.unlink()
        else:
            raise RuntimeError(
                f"Path {socket_path} exists but is not a socket; refusing to remove"
            )

    server.add_insecure_port(f"unix:{socket_path}")

    logger.info("Starting extraction service on %s", socket_path)
    await server.start()

    # Start idle monitor
    monitor = asyncio.create_task(idle_monitor(servicer, config))

    # Handle shutdown signals
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("Extraction service ready")
    await stop_event.wait()

    logger.info("Shutting down extraction service...")
    monitor.cancel()
    try:
        await monitor
    except asyncio.CancelledError:
        pass
    await server.stop(grace=5)
    servicer.unload_models()

    # Clean up socket
    if sock.exists() and stat.S_ISSOCK(sock.stat().st_mode):
        sock.unlink()

    logger.info("Extraction service stopped")


if __name__ == "__main__":
    asyncio.run(serve())
