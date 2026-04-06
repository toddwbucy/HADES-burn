"""LaTeX extraction backend — pure text parsing, no ML.

Handles .tex files and .tar.gz arXiv source packages.
Extracts equations, tables, citations, and section structure.
"""

from __future__ import annotations

import gzip
import logging
import re
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from .docling_backend import ExtractionResult

logger = logging.getLogger(__name__)


class LaTeXExtractor:
    """Extract structured content from LaTeX source files."""

    SUPPORTED_EXTENSIONS = {".tex", ".gz", ".tar.gz"}

    def extract(
        self,
        file_path: str | Path,
        *,
        extract_tables: bool = True,
        extract_equations: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        path = Path(file_path)
        start = time.time()

        if not path.exists():
            return ExtractionResult(error=f"File not found: {path}")

        suffix = path.suffix.lower()
        name = path.name.lower()

        try:
            if name.endswith(".tar.gz") or name.endswith(".tgz"):
                return self._extract_tar_gz(path, start, extract_tables, extract_equations)
            elif suffix == ".gz":
                return self._extract_plain_gz(path, start, extract_tables, extract_equations)
            elif suffix == ".tex":
                return self._extract_tex(path, start, extract_tables, extract_equations)
            else:
                return ExtractionResult(error=f"Unsupported LaTeX format: {suffix}")
        except Exception as e:
            return ExtractionResult(
                error=f"LaTeX extraction failed: {e}",
                processing_time=time.time() - start,
            )

    def _extract_tar_gz(self, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Extract from arXiv .tar.gz source package."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with tarfile.open(path, "r:gz") as tar:
                # Security: filter members
                safe = [
                    m for m in tar.getmembers()
                    if (m.isreg() or m.isdir()) and self._is_safe(m, tmp_path)
                ]
                tar.extractall(tmp_path, members=safe)

            # Find the main .tex file (largest)
            tex_files = list(tmp_path.rglob("*.tex"))
            if not tex_files:
                return ExtractionResult(
                    error="No .tex files found in archive",
                    processing_time=time.time() - start,
                )

            main_tex = max(tex_files, key=lambda f: f.stat().st_size)
            latex = main_tex.read_text(encoding="utf-8", errors="replace")

            return self._build_result(latex, path, start, extract_tables, extract_equations)

    def _extract_plain_gz(self, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Extract from gzipped .tex file."""
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            latex = f.read()
        return self._build_result(latex, path, start, extract_tables, extract_equations)

    def _extract_tex(self, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Extract from plain .tex file."""
        latex = path.read_text(encoding="utf-8", errors="replace")
        return self._build_result(latex, path, start, extract_tables, extract_equations)

    def _build_result(self, latex: str, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Build structured result from LaTeX source."""
        # Strip LaTeX commands for plain text (rough)
        text = self._strip_commands(latex)

        equations = self._extract_equations(latex) if extract_equations else []
        tables = self._extract_tables(latex) if extract_tables else []
        sections = self._extract_sections(latex)

        metadata = {
            "source": str(path),
            "format": "latex",
            "extractor": "latex_native",
            "num_equations": str(len(equations)),
            "num_tables": str(len(tables)),
            "num_sections": str(len(sections)),
        }

        return ExtractionResult(
            text=text,
            equations=equations,
            tables=tables,
            metadata=metadata,
            processing_time=time.time() - start,
        )

    @staticmethod
    def _is_safe(member: tarfile.TarInfo, target: Path) -> bool:
        """Reject symlinks, absolute paths, path traversal."""
        if member.issym() or member.islnk():
            return False
        if member.name.startswith("/") or ":" in member.name[:3]:
            return False
        target_resolved = target.resolve()
        resolved = (target / member.name).resolve()
        try:
            resolved.relative_to(target_resolved)
            return True
        except ValueError:
            return False

    @staticmethod
    def _strip_commands(latex: str) -> str:
        """Rough LaTeX → plain text (for full_text field)."""
        text = re.sub(r"\\begin\{.*?\}", "", latex)
        text = re.sub(r"\\end\{.*?\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})?", "", text)
        text = re.sub(r"[{}]", "", text)
        text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_equations(latex: str) -> list[dict[str, Any]]:
        """Extract equations from LaTeX source."""
        equations: list[dict[str, Any]] = []
        idx = 0

        # Display equations: \[ ... \], equation, align, etc.
        for env in ("equation", "equation*", "align", "align*", "gather", "gather*"):
            for m in re.finditer(
                rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}",
                latex,
                re.DOTALL,
            ):
                equations.append({
                    "latex": m.group(1).strip(),
                    "text": "",
                    "index": idx,
                    "is_inline": False,
                })
                idx += 1

        # Bracket display math: \[ ... \]
        for m in re.finditer(r"\\\[(.*?)\\\]", latex, re.DOTALL):
            equations.append({
                "latex": m.group(1).strip(),
                "text": "",
                "index": idx,
                "is_inline": False,
            })
            idx += 1

        # Inline math: \( ... \)
        for m in re.finditer(r"\\\((.*?)\\\)", latex, re.DOTALL):
            equations.append({
                "latex": m.group(1).strip(),
                "text": "",
                "index": idx,
                "is_inline": True,
            })
            idx += 1

        # Inline math: $...$ (but not $$...$$, and not escaped \$)
        for m in re.finditer(r"(?<!\$)(?<!\\)\$((?!\$).+?)\$(?!\$)", latex):
            equations.append({
                "latex": m.group(1).strip(),
                "text": "",
                "index": idx,
                "is_inline": True,
            })
            idx += 1

        return equations

    @classmethod
    def _extract_tables(cls, latex: str) -> list[dict[str, Any]]:
        """Extract tables from LaTeX source."""
        tables: list[dict[str, Any]] = []

        for i, m in enumerate(
            re.finditer(
                r"\\begin\{table\}(.*?)\\end\{table\}",
                latex,
                re.DOTALL,
            )
        ):
            content = m.group(1)
            caption = cls._extract_braced(content, r"\caption")

            tables.append({
                "content": content.strip(),
                "caption": caption,
                "index": i,
            })

        return tables

    @classmethod
    def _extract_sections(cls, latex: str) -> list[dict[str, Any]]:
        """Extract section headings."""
        sections: list[dict[str, Any]] = []
        level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}

        for m in re.finditer(
            r"\\(section|subsection|subsubsection|paragraph)\*?\{",
            latex,
        ):
            cmd = m.group(1)
            title = cls._extract_braced(latex[m.start():], f"\\{cmd}")
            if not title:
                title = cls._extract_braced(latex[m.start():], f"\\{cmd}*")
            sections.append({
                "level": level_map.get(cmd, 0),
                "title": title,
            })

        return sections

    @staticmethod
    def _extract_braced(text: str, command: str) -> str:
        """Extract brace-delimited argument from a LaTeX command, handling nesting."""
        idx = text.find(command)
        if idx == -1:
            return ""
        # Find the opening brace after the command
        brace_start = text.find("{", idx + len(command))
        if brace_start == -1:
            return ""
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1 : i]
        return ""
