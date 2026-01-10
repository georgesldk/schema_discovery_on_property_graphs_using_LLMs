import re


def strip_comments(text):
    """
    Remove single-line (//) and multi-line (/* */) comments from text.

    Logic moved 1:1 from extract_gt.py.
    """
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text
