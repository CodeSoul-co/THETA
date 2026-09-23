from __future__ import annotations

import csv
import codecs
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .errors import PermanentJobError


SUPPORTED_FORMATS = frozenset({"csv", "tsv", "txt", "md", "json", "jsonl", "ndjson", "xlsx", "xls", "parquet", "pdf", "docx"})
INPUT_PREFIX = "input."


def split_parameters(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate worker-owned input mappings from engine parameters."""
    input_options = {
        name.removeprefix(INPUT_PREFIX): value
        for name, value in params.items()
        if name.startswith(INPUT_PREFIX)
    }
    engine_params = {
        name: value for name, value in params.items() if not name.startswith(INPUT_PREFIX)
    }
    return input_options, engine_params


def normalize_dataset(source: Path, destination: Path, file_format: str, options: dict[str, Any]) -> None:
    normalized_format = (file_format or source.suffix.lstrip(".")).strip().lower()
    if normalized_format == "ndjson":
        normalized_format = "jsonl"
    if normalized_format not in SUPPORTED_FORMATS:
        raise PermanentJobError(
            f"unsupported dataset format {normalized_format!r}; supported: {sorted(SUPPORTED_FORMATS)}"
        )

    rows = _read_rows(source, normalized_format, options.get("sheet"))
    try:
        first = next(rows)
    except StopIteration as exc:
        raise PermanentJobError("dataset does not contain any records") from exc

    columns = list(first)
    text_column = _select_text_column(columns, options.get("text_column"))
    normalized_input = text_column == "text"
    label_column = _optional_column(
        columns,
        options.get("label_column") if "label_column" in options else ("label" if normalized_input and "label" in columns else None),
        "label_column",
    )
    time_column = _optional_column(
        columns,
        options.get("time_column") if "time_column" in options else _normalized_time_column(columns, normalized_input),
        "time_column",
    )
    covariates = options.get("covariates") if "covariates" in options else _normalized_covariates(columns, normalized_input)
    if not isinstance(covariates, list) or not all(isinstance(item, str) and item for item in covariates):
        raise PermanentJobError("input.covariates must be an array of non-empty column names")
    unknown_covariates = [column for column in covariates if column not in columns]
    if unknown_covariates:
        raise PermanentJobError(f"covariate columns were not found: {unknown_covariates}")

    fieldnames = ["text"]
    if label_column:
        fieldnames.append("label")
    if time_column:
        fieldnames.extend(["year", "timestamp"])
    fieldnames.extend(f"cov_{index}" for index in range(len(covariates)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in _prepend(first, rows):
                text = _scalar(row.get(text_column))
                if not text.strip():
                    continue
                output: dict[str, str] = {"text": text}
                if label_column:
                    output["label"] = _scalar(row.get(label_column))
                if time_column:
                    time_value = _scalar(row.get(time_column))
                    output["year"] = time_value
                    output["timestamp"] = time_value
                for index, column in enumerate(covariates):
                    output[f"cov_{index}"] = _scalar(row.get(column))
                writer.writerow(output)
                count += 1
        if count == 0:
            raise PermanentJobError("dataset has no non-empty values in the selected text column")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rows(source: Path, file_format: str, sheet: Any) -> Iterable[dict[str, Any]]:
    if file_format in {"csv", "tsv"}:
        encoding = _encoding(source)
        delimiter = "\t" if file_format == "tsv" else _delimiter(source, encoding)
        with source.open("r", encoding=encoding, errors="strict", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
        return
    if file_format in {"txt", "md"}:
        with source.open("r", encoding=_encoding(source), errors="strict") as handle:
            for line in handle:
                if line.strip():
                    yield {"text": line.strip()}
        return
    if file_format == "pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise PermanentJobError("reading pdf requires PyPDF2") from exc
        try:
            for index, page in enumerate(PdfReader(str(source)).pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    yield {"text": text, "page": index}
        except Exception as exc:
            raise PermanentJobError(f"could not extract PDF text: {exc}") from exc
        return
    if file_format == "docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise PermanentJobError("reading docx requires python-docx") from exc
        try:
            for index, paragraph in enumerate(Document(str(source)).paragraphs, start=1):
                text = paragraph.text.strip()
                if text:
                    yield {"text": text, "paragraph": index}
        except Exception as exc:
            raise PermanentJobError(f"could not extract DOCX text: {exc}") from exc
        return
    if file_format == "jsonl":
        with source.open("r", encoding=_encoding(source), errors="strict") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PermanentJobError("JSONL records must be objects")
                yield value
        return
    if file_format == "json":
        with source.open("r", encoding=_encoding(source), errors="strict") as handle:
            value = json.load(handle)
        records = value
        if isinstance(value, dict):
            records = value.get("data", value.get("records", value.get("items")))
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise PermanentJobError("JSON dataset must be an object array or contain data/records/items")
        yield from records
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise PermanentJobError(f"reading {file_format} requires pandas and its format adapter") from exc
    try:
        if file_format == "parquet":
            frame = pd.read_parquet(source)
        else:
            with pd.ExcelFile(source) as workbook:
                selected = sheet
                if selected is not None and selected not in workbook.sheet_names:
                    raise PermanentJobError(
                        f"worksheet {selected!r} was not found; available: {workbook.sheet_names}"
                    )
                if selected is None:
                    selected = next(
                        (name for name in workbook.sheet_names if not pd.read_excel(workbook, sheet_name=name, nrows=1).empty),
                        workbook.sheet_names[0] if workbook.sheet_names else None,
                    )
                if selected is None:
                    raise PermanentJobError("workbook does not contain any worksheets")
                frame = pd.read_excel(workbook, sheet_name=selected)
    except PermanentJobError:
        raise
    except Exception as exc:
        raise PermanentJobError(f"could not read {file_format} dataset: {exc}") from exc
    frame = frame.where(frame.notna(), None)
    yield from frame.to_dict(orient="records")


def _select_text_column(columns: list[str], requested: Any) -> str:
    if requested is not None:
        if not isinstance(requested, str) or not requested:
            raise PermanentJobError("input.text_column must be a non-empty column name")
        if requested not in columns:
            raise PermanentJobError(f"text column {requested!r} was not found; available: {columns}")
        return requested
    for candidate in ("text", "cleaned_content", "content", "body", "document", "文本", "内容"):
        if candidate in columns:
            return candidate
    if len(columns) == 1:
        return columns[0]
    raise PermanentJobError("input.text_column is required when the dataset has no conventional text column")


def _optional_column(columns: list[str], value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PermanentJobError(f"input.{name} must be a non-empty column name")
    if value not in columns:
        raise PermanentJobError(f"{name} {value!r} was not found; available: {columns}")
    return value


def _normalized_time_column(columns: list[str], normalized_input: bool) -> str | None:
    if not normalized_input:
        return None
    for candidate in ("timestamp", "year"):
        if candidate in columns:
            return candidate
    return None


def _normalized_covariates(columns: list[str], normalized_input: bool) -> list[str]:
    if not normalized_input:
        return []
    covariates = [column for column in columns if re.fullmatch(r"cov_\d+", column)]
    return sorted(covariates, key=lambda column: int(column.removeprefix("cov_")))


def _encoding(path: Path) -> str:
    sample = path.read_bytes()[:65536]
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if sample.startswith(b"\xfe\xff"):
        return "utf-16-be"
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            # The bounded probe can end inside a multibyte character. An
            # incremental decoder validates every complete sequence while
            # retaining an incomplete trailing sequence for the next chunk.
            codecs.getincrementaldecoder(encoding)(errors="strict").decode(sample, final=False)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _delimiter(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="strict") as handle:
        sample = "".join(handle.readline() for _ in range(20))
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return ","


def _prepend(first: dict[str, Any], rest: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    yield first
    yield from rest


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)
