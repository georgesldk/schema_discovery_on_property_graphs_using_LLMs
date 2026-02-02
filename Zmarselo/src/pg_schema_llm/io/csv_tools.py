from __future__ import annotations

import csv
from typing import Iterable, List, Sequence

import pandas as pd

CSV_SAMPLE_BYTES = 4096


def sniff_delimiter(file_path: str, default: str = ",") -> str:
    """
    Detect the delimiter used in a CSV file.

    This function attempts to infer the delimiter by inspecting a small
    sample of the file. It handles common delimiters and falls back to
    a default value when detection fails or the file cannot be read.

    Args:
        file_path (str): Path to the CSV file.
        default (str): Delimiter to return if detection fails.

    Returns:
        str: Detected delimiter character.
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
    Normalize CSV header column names for robust file-role detection.

    This function standardizes column headers by stripping quotes and
    normalizing common identifier variants (e.g., ':ID', 'ID') to a
    canonical form. Special Neo4j headers such as ':START_ID' and
    ':END_ID' are preserved.

    Args:
        cols (Sequence[str]): Raw column names from the CSV header.

    Returns:
        List[str]: Normalized column names.
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
    Read a lightweight preview of a CSV file.

    This function loads only the header row of a CSV file in order to
    inspect column names without incurring the cost of a full read.
    It is used during file-role detection and dataset profiling.

    Args:
        file_path (str): Path to the CSV file.
        delim (str): Column delimiter.

    Returns:
        pd.DataFrame: Empty dataframe containing only normalized columns.
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
    Read an entire CSV file into a dataframe.

    This function performs a full CSV load with robust parsing options
    and normalized headers. It is primarily intended for legacy graph
    construction paths where streaming is not required.

    Args:
        file_path (str): Path to the CSV file.
        delim (str): Column delimiter.

    Returns:
        pd.DataFrame: Fully loaded dataframe with normalized columns.
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
    Iterate over a CSV file in chunks with robust parsing.

    This generator yields dataframe chunks to enable streaming
    processing of large datasets. It avoids materializing the full
    file in memory and stabilizes parsing for heterogeneous or
    messy input files.

    Args:
        file_path (str): Path to the CSV file.
        delim (str): Column delimiter.
        chunksize (int): Number of rows per chunk.
        dtype_str (bool): Whether to force all columns to string type.
        on_bad_lines (str): Policy for handling malformed rows.

    Returns:
        Iterable[pd.DataFrame]: Iterator over dataframe chunks.
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
