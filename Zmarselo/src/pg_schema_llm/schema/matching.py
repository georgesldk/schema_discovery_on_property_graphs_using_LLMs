from difflib import SequenceMatcher


def similar(a, b):
    """
    Compute a case-insensitive similarity score between two strings.

    This function uses sequence matching to quantify how similar two
    strings are after normalizing them to lowercase.

    Args:
        a (str): First string.
        b (str): Second string.

    Returns:
        float: Similarity score in the range [0.0, 1.0].
    """

    if not a or not b:
        return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(target, candidates, threshold=0.8):
    """
    Select the most similar candidate string above a threshold.

    This function compares a target string against a collection of
    candidate strings and returns the best match whose similarity
    score exceeds the specified threshold.

    Args:
        target (str): Target string to match.
        candidates (Iterable[str]): Candidate strings.
        threshold (float): Minimum similarity score for acceptance.

    Returns:
        Optional[str]: Best matching candidate, or None if no candidate
        satisfies the threshold.
    """

    best = None
    best_score = 0
    for c in candidates:
        s = similar(target, c)
        if s > best_score:
            best_score = s
            best = c
    return best if best_score >= threshold else None
