from __future__ import annotations

import csv
from typing import Iterable, List, Sequence

import pandas as pd

CSV_SAMPLE_BYTES = 4096


def sniff_delimiter(file_path: str, default: str = ",") -> str:
    """
    Detect delimiter robustly.
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(CSV_SAMPLE_BYTES)
    except Exception:
        return default

    first_line = sample.splitlines()[0] if sample else ""
    if "|" in first_line and "," not in first_line:
        return "|"

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        return default


def normalize_header_columns(cols: Sequence[str]) -> List[str]:
    """
    Normalize column names so detect_file_role works across formats:
    - strip surrounding quotes
    - map ':ID' / 'ID' / 'id' -> 'id' (only for header convenience)
    NOTE: We DO NOT remove ':START_ID' or ':END_ID' because detect_file_role uses them.
    """
    out: List[str] = []
    for c in cols:
        c2 = str(c).strip().strip('"').strip("'")
        if c2 in (":ID", "ID", "id"):
            c2 = "id"
        out.append(c2)
    return out


def read_preview(file_path: str, delim: str) -> pd.DataFrame:
    """
    Read headers only (robust to quotes).
    """
    df = pd.read_csv(
        file_path,
        nrows=0,
        sep=delim,
        engine="python",
        quotechar='"',
        doublequote=True,
    )
    df.columns = normalize_header_columns(df.columns)
    return df


def read_full_df(file_path: str, delim: str) -> pd.DataFrame:
    """
    Full read (legacy graph builder).
    """
    df = pd.read_csv(
        file_path,
        sep=delim,
        engine="python",
        quotechar='"',
        doublequote=True,
        dtype=str,
        keep_default_na=False,
        na_values=["", "null", "NULL"],
        on_bad_lines="skip",
    )
    df.columns = normalize_header_columns(df.columns)
    return df


def iter_chunks(
    file_path: str,
    delim: str,
    chunksize: int,
    *,
    dtype_str: bool = True,
    on_bad_lines: str = "skip",
) -> Iterable[pd.DataFrame]:
    """
    Chunk iterator with robust parsing options.
    dtype_str=True makes reading stable for mixed-type columns / messy files.
    """
    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        sep=delim,
        engine="python",
        quotechar='"',
        doublequote=True,
        dtype=str if dtype_str else None,
        keep_default_na=False,
        na_values=["", "null", "NULL"],
        on_bad_lines=on_bad_lines,
    ):
        chunk.columns = normalize_header_columns(chunk.columns)
        yield chunk
