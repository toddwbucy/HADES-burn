"""Docling extraction backend — the ML boundary for document processing.

Wraps docling's DocumentConverter behind a minimal interface.
Everything else (batching, chunking, embedding, DB writes) is handled
by the Rust orchestrator.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured extraction output matching the gRPC ExtractResponse."""

    text: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    processing_time: float = 0.0


class DoclingExtractor:
    """Extracts structured content from PDF documents using Docling VLM.

    Lazy-loads the docling converter on first use. Call cleanup() to
    release GPU memory when the model is no longer needed.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".text", ".md"}

    def __init__(self, *, use_ocr: bool = False, use_fallback: bool = True) -> None:
        self._converter = None
        self._converter_lock = threading.Lock()
        self._use_ocr = use_ocr
        self._use_fallback = use_fallback

    @property
    def converter(self):
        """Lazy-load the Docling document converter (thread-safe)."""
        if self._converter is None:
            with self._converter_lock:
                if self._converter is not None:
                    return self._converter

                logger.info("Loading Docling document converter...")
                start = time.time()
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                    TableStructureOptions,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption

                pdf_options = PdfPipelineOptions(
                    do_table_structure=True,
                    do_ocr=self._use_ocr,
                    table_structure_options=TableStructureOptions(do_cell_matching=True),
                )
                self._converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
                )
                logger.info("Docling converter loaded in %.2fs", time.time() - start)
        return self._converter

    def extract(
        self,
        file_path: str | Path,
        *,
        extract_tables: bool = True,
        extract_equations: bool = True,
        extract_images: bool = True,
        use_ocr: bool | None = None,
    ) -> ExtractionResult:
        """Extract structured content from a document.

        This is the only method that touches ML models.
        """
        path = Path(file_path)
        start = time.time()
        effective_ocr = use_ocr if use_ocr is not None else self._use_ocr

        if not path.exists():
            return ExtractionResult(error=f"File not found: {path}")

        if not path.is_file():
            return ExtractionResult(error=f"Not a file: {path}")

        # Plain text fast path
        if path.suffix.lower() in (".txt", ".text", ".md"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                return ExtractionResult(
                    text=text,
                    metadata={"source": str(path), "format": path.suffix.lstrip(".")},
                    processing_time=time.time() - start,
                )
            except Exception as e:
                return ExtractionResult(error=f"Failed to read text file: {e}")

        # PDF extraction via Docling
        if path.suffix.lower() == ".pdf":
            if effective_ocr != self._use_ocr:
                logger.warning(
                    "Per-call use_ocr=%s differs from converter default=%s; "
                    "converter OCR setting cannot be changed at call time",
                    effective_ocr, self._use_ocr,
                )
            return self._extract_pdf(path, start, extract_tables, extract_equations, extract_images)

        return ExtractionResult(error=f"Unsupported format: {path.suffix}")

    def _extract_pdf(
        self,
        path: Path,
        start: float,
        extract_tables: bool,
        extract_equations: bool,
        extract_images: bool,
    ) -> ExtractionResult:
        """Extract from PDF using Docling, with optional PyMuPDF fallback."""
        # Validate PDF header
        try:
            with open(path, "rb") as f:
                header = f.read(5)
            if header != b"%PDF-":
                return ExtractionResult(error=f"Invalid PDF header: {header!r}")
        except Exception as e:
            return ExtractionResult(error=f"Cannot read file: {e}")

        try:
            return self._extract_with_docling(path, start, extract_tables, extract_equations, extract_images)
        except Exception as e:
            logger.error("Docling extraction failed for %s: %s", path, e)
            if self._use_fallback:
                logger.info("Attempting PyMuPDF fallback for %s", path)
                return self._extract_fallback(path, start)
            return ExtractionResult(
                error=f"Docling extraction failed: {e}",
                processing_time=time.time() - start,
            )

    def _extract_with_docling(
        self,
        path: Path,
        start: float,
        extract_tables: bool,
        extract_equations: bool,
        extract_images: bool,
    ) -> ExtractionResult:
        """Core Docling extraction �� THE ML CALL."""
        result = self.converter.convert_single(str(path))

        # Docling v2 exposes the document via result.document or result.output
        doc = getattr(result, "document", None) or getattr(result, "output", None)

        # Extract markdown as the full text
        full_text = ""
        if doc is not None and hasattr(doc, "export_to_markdown"):
            full_text = doc.export_to_markdown()

        tables = []
        equations = []
        images = []

        if extract_tables and doc is not None:
            for i, table in enumerate(getattr(doc, "tables", [])):
                tables.append({
                    "content": table.export_to_markdown() if hasattr(table, "export_to_markdown") else str(table),
                    "caption": getattr(table, "caption", ""),
                    "index": i,
                })

        if extract_equations and doc is not None:
            # Docling v2: try doc.equations first, fall back to scanning doc.texts
            eq_items = getattr(doc, "equations", []) or []
            for i, item in enumerate(eq_items):
                equations.append({
                    "latex": getattr(item, "text", str(item)),
                    "text": getattr(item, "text", ""),
                    "index": i,
                    "is_inline": getattr(item, "is_inline", False),
                })
            # If no equations found via doc.equations, scan texts for formula entries
            if not equations:
                for item in getattr(doc, "texts", []):
                    obj_type = getattr(item, "obj_type", "")
                    if obj_type == "equation" or hasattr(item, "latex"):
                        equations.append({
                            "latex": getattr(item, "latex", getattr(item, "text", str(item))),
                            "text": getattr(item, "text", ""),
                            "index": len(equations),
                            "is_inline": getattr(item, "is_inline", False),
                        })

        if extract_images and doc is not None:
            for i, fig in enumerate(getattr(doc, "pictures", [])):
                images.append({
                    "path": getattr(fig, "uri", ""),
                    "caption": getattr(fig, "caption", ""),
                    "index": i,
                })

        metadata = {
            "source": str(path),
            "format": "pdf",
            "extractor": "docling",
            "num_pages": str(getattr(doc, "num_pages", 0)) if doc is not None else "0",
        }

        return ExtractionResult(
            text=full_text,
            tables=tables,
            equations=equations,
            images=images,
            metadata=metadata,
            processing_time=time.time() - start,
        )

    def _extract_fallback(self, path: Path, start: float) -> ExtractionResult:
        """Fallback extraction using PyMuPDF (no ML, text-only)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ExtractionResult(
                error="Docling failed and PyMuPDF not available for fallback",
                processing_time=time.time() - start,
            )

        try:
            with fitz.open(str(path)) as doc:
                pages = [page.get_text() for page in doc]
            full_text = "\n\n".join(pages)

            return ExtractionResult(
                text=full_text,
                metadata={
                    "source": str(path),
                    "format": "pdf",
                    "extractor": "pymupdf_fallback",
                    "num_pages": str(len(pages)),
                },
                processing_time=time.time() - start,
            )
        except Exception as e:
            return ExtractionResult(
                error=f"Fallback extraction failed: {e}",
                processing_time=time.time() - start,
            )

    def cleanup(self) -> None:
        """Release GPU memory held by the Docling converter."""
        if self._converter is not None:
            logger.info("Unloading Docling converter...")
            del self._converter
            self._converter = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except ImportError:
                pass
            logger.info("Docling converter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._converter is not None
