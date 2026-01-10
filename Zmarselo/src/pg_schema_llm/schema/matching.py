from difflib import SequenceMatcher


def similar(a, b):
    if not a or not b:
        return 0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(target, candidates, threshold=0.8):
    best = None
    best_score = 0
    for c in candidates:
        s = similar(target, c)
        if s > best_score:
            best_score = s
            best = c
    return best if best_score >= threshold else None
