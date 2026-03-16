from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class CandidateSummary:
    batch_name: str
    image_label: str
    image_number: int
    vote_count: int
    reason_summary: str
    winner: bool


def load_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root:
        texts = [node.text or "" for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
        values.append("".join(texts))
    return values


def col_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - 64)
    return value - 1


def parse_sheet_rows(workbook: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str]]:
    root = ET.fromstring(workbook.read(sheet_path))
    rows = root.find("main:sheetData", NS)
    if rows is None:
        return []

    parsed_rows: list[list[str]] = []
    for row in rows:
        values: dict[int, str] = {}
        max_col = -1
        for cell in row:
            idx = col_to_index(cell.attrib.get("r", "A1"))
            max_col = max(max_col, idx)
            value = ""

            node = cell.find("main:v", NS)
            if node is not None:
                value = node.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]

            inline = cell.find("main:is", NS)
            if inline is not None:
                texts = [n.text or "" for n in inline.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                value = "".join(texts)

            values[idx] = value.strip()

        if max_col >= 0:
            parsed_rows.append([values.get(i, "").strip() for i in range(max_col + 1)])
    return parsed_rows


def extract_sheet_rows(workbook_path: Path) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(workbook_path, "r") as workbook:
        shared_strings = load_shared_strings(workbook)
        rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_map = {node.attrib["Id"]: f"xl/{node.attrib['Target']}" for node in rel_root}
        wb_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheets = wb_root.find("main:sheets", NS)
        if sheets is None:
            return {}

        result: dict[str, list[list[str]]] = {}
        for sheet in sheets:
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            result[sheet.attrib["name"]] = parse_sheet_rows(workbook, rel_map[rel_id], shared_strings)
        return result


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except ValueError:
        return 0


def extract_image_number(label: str) -> int:
    match = re.search(r"(\d+)$", label)
    return int(match.group(1)) if match else 0


def looks_like_candidate_row(row: list[str]) -> bool:
    if len(row) < 2:
        return False
    if not row[0] or extract_image_number(row[0]) == 0:
        return False
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", row[1]))


def parse_candidate_rows(sheet_name: str, rows: list[list[str]]) -> list[CandidateSummary]:
    candidates: list[CandidateSummary] = []
    max_votes = -1
    for row in rows:
        if not looks_like_candidate_row(row):
            continue
        image_number = extract_image_number(row[0])
        vote_count = parse_int(row[1])
        reason_summary = row[2] if len(row) > 2 else ""
        candidates.append(
            CandidateSummary(
                batch_name=sheet_name,
                image_label=row[0],
                image_number=image_number,
                vote_count=vote_count,
                reason_summary=reason_summary,
                winner=False,
            )
        )
        max_votes = max(max_votes, vote_count)

    for candidate in candidates:
        candidate.winner = candidate.vote_count == max_votes and max_votes >= 0
    return candidates


def extract_embedded_images(workbook_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(workbook_path, "r") as workbook:
        for member in workbook.namelist():
            if not member.startswith("xl/media/"):
                continue
            target = output_dir / Path(member).name
            with workbook.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            extracted.append(str(target))
    return extracted


def build_dataset(workbook_path: Path, output_path: Path, image_output_dir: Path) -> dict[str, object]:
    rows_by_sheet = extract_sheet_rows(workbook_path)
    records: list[dict[str, object]] = []
    for sheet_name, rows in rows_by_sheet.items():
        candidates = parse_candidate_rows(sheet_name, rows)
        if not candidates:
            continue
        records.append(
            {
                "batch_name": sheet_name,
                "candidate_count": len(candidates),
                "winner_labels": [candidate.image_label for candidate in candidates if candidate.winner],
                "winner_indices": [candidate.image_number for candidate in candidates if candidate.winner],
                "candidates": [asdict(candidate) for candidate in candidates],
            }
        )

    extracted_images = extract_embedded_images(workbook_path, image_output_dir)
    payload = {
        "source_file": str(workbook_path),
        "record_count": len(records),
        "records": records,
        "extracted_media": extracted_images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    workspace = Path(__file__).resolve().parent
    workbook_path = workspace / "input.xlsx"
    output_path = workspace / "data" / "processed" / "training_dataset.json"
    image_output_dir = workspace / "data" / "processed" / "embedded_media"

    payload = build_dataset(workbook_path, output_path, image_output_dir)
    print(f"records={payload['record_count']}")
    print(f"embedded_media={len(payload['extracted_media'])}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
