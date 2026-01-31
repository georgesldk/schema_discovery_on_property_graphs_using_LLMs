from __future__ import annotations

import os
import re
from typing import Sequence, Tuple


def get_common_affixes(filenames: Sequence[str]) -> Tuple[str, str]:
    """
    Dataset-agnostic noise detector: common prefix/suffix across *all* filenames.
    Works best when most files share a wrapper like 'mb6_' or '_edges'.
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
    Remove role tokens only if they appear as whole tokens at the start/end.
    This avoids assumptions like nodes_/rels_ and avoids damaging names like 'Knowledge'.
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
    Fully dataset-agnostic type name cleaner:
    - removes global common prefix/suffix across files
    - strips extension
    - strips leading/trailing separators
    - removes generic role tokens only when standalone tokens at the start/end

    Examples (no hardcoded 'nodes_'/'rels_'):
      "mb6_nodes_Person.csv"      -> "Person"
      "FIB25-edges-PURCHASED.csv" -> "PURCHASED"
      "relationships_user_tag.csv"-> "user_tag"
      "node.Person.csv"           -> "Person"
      "KnowledgeGraph.csv"        -> "KnowledgeGraph"   (unchanged, no token match)
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
    return os.path.splitext(os.path.basename(filename))[0]
