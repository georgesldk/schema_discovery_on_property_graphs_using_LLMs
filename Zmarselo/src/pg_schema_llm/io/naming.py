from __future__ import annotations

import os
import re
from typing import Sequence, Tuple


def get_common_affixes(filenames: Sequence[str]) -> Tuple[str, str]:
    """
    Detect common filename prefix and suffix across a dataset.

    This function identifies shared leading and trailing substrings
    present in all filenames. These affixes typically represent
    dataset-level wrappers or noise (e.g., dataset identifiers or
    role indicators) and can be safely removed during name normalization.

    Args:
        filenames (Sequence[str]): Collection of filenames.

    Returns:
        Tuple[str, str]: Common (prefix, suffix) shared by all filenames.
    """
    if not filenames:
        return "", ""
    prefix = os.path.commonprefix(list(filenames))
    rev = [f[::-1] for f in filenames]
    suffix = os.path.commonprefix(rev)[::-1]
    return prefix, suffix


# generic, role-ish tokens (only removed when they are standalone tokens)
_ROLE_TOKENS = {
    "node", "nodes", "vertex", "vertices", "entity", "entities",
    "edge", "edges", "relationship", "relationships", "relation", "relations",
    "rel", "rels", "link", "links",
}

# separators we consider ?token boundaries? in filenames
_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)  # splits on _ - . space etc.


def _strip_role_tokens(tokens: Sequence[str]) -> Tuple[str, ...]:
    """
    Remove generic role tokens from tokenized filenames.

    This function removes role-related tokens (e.g., 'nodes', 'edges')
    only when they appear as complete tokens at the beginning or end
    of the token sequence. This conservative strategy avoids accidental
    modification of meaningful type names.

    Args:
        tokens (Sequence[str]): Tokenized filename components.

    Returns:
        Tuple[str, ...]: Tokens with role indicators removed from boundaries.
    """
    toks = [t for t in tokens if t]  # drop empties
    if not toks:
        return tuple()

    # trim from front
    while toks and toks[0].lower() in _ROLE_TOKENS:
        toks = toks[1:]

    # trim from end
    while toks and toks[-1].lower() in _ROLE_TOKENS:
        toks = toks[:-1]

    return tuple(toks)


def clean_name_smart(filename: str, prefix: str, suffix: str) -> str:
    """
    Derive a clean, dataset-agnostic type name from a filename.

    This function removes dataset-level noise such as shared prefixes
    and suffixes, file extensions, separator artifacts, and generic
    role tokens. It is designed to operate without hardcoded dataset
    assumptions and preserves semantic identifiers whenever possible.

    Args:
        filename (str): Original filename.
        prefix (str): Common dataset prefix to remove.
        suffix (str): Common dataset suffix to remove.

    Returns:
        str: Normalized type name suitable for schema inference.
    """
    base = os.path.basename(filename)

    # remove global common prefix/suffix (safe, dataset-agnostic)
    if prefix and base.startswith(prefix):
        base = base[len(prefix):]
    if suffix and base.endswith(suffix):
        base = base[:-len(suffix)]

    # drop extension + trim separators
    stem = os.path.splitext(base)[0].strip(" _-.\t")

    if not stem:
        return ""

    # split into tokens by separators and trim generic role tokens at boundaries
    toks = _TOKEN_SPLIT_RE.split(stem)
    stripped = _strip_role_tokens(toks)

    if stripped:
        # rebuild using underscore for stability (doesn't assume original separators)
        return "_".join(stripped).strip(" _-.\t")

    # if stripping removed everything, fall back to original stem
    return stem


# --- Legacy support ---
def clean_type_name(filename: str) -> str:
    """
    Extract a basic type name from a filename (legacy behavior).

    This function strips directory paths and file extensions without
    performing any dataset-agnostic normalization. It is retained for
    backward compatibility with earlier code paths.

    Args:
        filename (str): Original filename.

    Returns:
        str: Filename stem without extension.
    """
    return os.path.splitext(os.path.basename(filename))[0]
