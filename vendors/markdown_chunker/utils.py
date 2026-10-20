"""
Utility functions for markdown chunking.
"""

import re
import hashlib
from typing import List, Tuple, Set


def is_heading(line: str) -> bool:
    """Check if a line is a heading."""
    return bool(re.match(r"^#{1,6}\s+", line.strip()))


def get_heading_level(line: str) -> int:
    """Get the level of a heading (1-6). Returns 0 if not a heading."""
    if not is_heading(line):
        return 0
    match = re.match(r"^(#{1,6})\s+", line.strip())
    if match:
        return len(match.group(1))
    return 0


def is_table_row(line: str) -> bool:
    """Check if a line is a markdown table row."""
    return bool(re.match(r"^\|.*\|$", line.strip()))


def is_table_separator(line: str) -> bool:
    """Check if a line is a markdown table separator."""
    return bool(re.match(r"^\|(\s*[-:]+\s*\|)+$", line.strip()))


def is_code_block_delimiter(line: str) -> bool:
    """Check if a line is a code block delimiter."""
    return bool(re.match(r"^```\w*$", line.strip())) or bool(
        re.match(r"^```$", line.strip())
    )


def is_list_item(line: str) -> bool:
    """Check if a line is a list item."""
    return bool(
        re.match(r"^\s*[-*+]\s+.+$", line.strip())
        or re.match(r"^\s*\d+\.\s+.+$", line.strip())
    )


def is_blockquote(line: str) -> bool:
    """Check if a line is a blockquote."""
    return bool(re.match(r"^\s*>\s*.*$", line.strip()))


def is_horizontal_rule(line: str) -> bool:
    """Check if a line is a horizontal rule."""
    return bool(re.match(r"^\s*[-*_]{3,}\s*$", line.strip()))


def is_yaml_delimiter(line: str) -> bool:
    """Check if a line is a YAML front matter delimiter."""
    return line.strip() == "---"


def is_html_tag(line: str) -> bool:
    """Check if a line contains an HTML tag."""
    return bool(re.search(r"<[^>]+>", line))


def is_footnote_def(line: str) -> bool:
    """Check if a line is a footnote definition."""
    return bool(re.match(r"^\[\^[^\]]+\]:\s+.*$", line.strip()))


def contains_footnote_ref(line: str) -> bool:
    """Check if a line contains a footnote reference."""
    return bool(re.search(r"\[\^[^\]]+\]", line))


def contains_image_or_link(line: str) -> bool:
    """Check if a line contains an image or link."""
    return bool(
        re.search(r"!\[.*?\]\(.*?\)", line) or re.search(r"\[.*?\]\(.*?\)", line)
    )


def compute_hash(text: str) -> str:
    """Compute a hash for text to identify duplicates."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def split_into_lines(markdown_text: str) -> List[str]:
    """Split markdown text into lines."""
    return markdown_text.splitlines()


def join_lines(lines: List[str]) -> str:
    """Join lines back into text."""
    return "\n".join(lines)


def find_sentence_boundaries(text: str) -> List[int]:
    """Find positions of sentence boundaries in text."""
    # This is a simplified implementation; a more robust one would handle
    # cases like "Dr. Smith" or "e.g." where periods don't end sentences
    boundaries = []
    for match in re.finditer(r"[.!?]\s+", text):
        boundaries.append(match.end())
    return boundaries


def find_logical_split_points(text: str, preferred_len: int) -> List[int]:
    """
    Find logical split points in text, preferring positions near preferred_len.
    Returns a list of character positions where the text can be split.
    """
    # Try to split at paragraph boundaries first
    paragraph_boundaries = []
    for match in re.finditer(r"\n\s*\n", text):
        paragraph_boundaries.append(match.end())

    # Try to split at sentence boundaries
    sentence_boundaries = find_sentence_boundaries(text)

    # Try to split at newline characters
    newline_boundaries = []
    for match in re.finditer(r"\n", text):
        if match.start() not in paragraph_boundaries:
            newline_boundaries.append(match.end())

    # Combine all boundaries and sort
    all_boundaries = sorted(
        set(paragraph_boundaries + sentence_boundaries + newline_boundaries)
    )

    # If no boundaries found, fall back to word boundaries
    if not all_boundaries:
        for match in re.finditer(r"\s+", text):
            all_boundaries.append(match.end())

    # Find the boundary closest to preferred_len
    best_boundaries = []
    if all_boundaries:
        best_boundary = min(all_boundaries, key=lambda x: abs(x - preferred_len))
        best_boundaries.append(best_boundary)

    return best_boundaries


def extract_footnote_refs(text: str) -> Set[str]:
    """Extract footnote references from text."""
    refs = set()
    for match in re.finditer(r"\[\^([^\]]+)\]", text):
        refs.add(match.group(1))
    return refs


def extract_footnote_defs(text: str) -> dict:
    """Extract footnote definitions from text."""
    defs = {}
    for match in re.finditer(r"\[\^([^\]]+)\]:\s+(.*?)($|\n\s*\n)", text, re.DOTALL):
        defs[match.group(1)] = match.group(2)
    return defs
