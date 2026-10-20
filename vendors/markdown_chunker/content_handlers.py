"""
Content handlers for different types of markdown elements.
"""

from typing import List
import re
from .utils import (
    is_heading,
    is_table_row,
    is_table_separator,
    is_code_block_delimiter,
    is_list_item,
    is_blockquote,
    is_horizontal_rule,
    is_html_tag,
    is_footnote_def,
    contains_footnote_ref,
    join_lines,
    split_into_lines,
)


class ContentHandler:
    """Base class for content handlers."""

    def can_handle(self, content: str) -> bool:
        """Check if this handler can handle the given content."""
        return False

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """Split the content if it exceeds max_len."""
        return [content]


class HeadingHandler(ContentHandler):
    """Handler for headings."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        return lines and is_heading(lines[0])

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Headings should not be split. If a heading with its content exceeds max_len,
        we'll keep the heading intact and split the content after it.
        """
        if len(content) <= max_len:
            return [content]

        lines = split_into_lines(content)
        heading_lines = []
        content_lines = []

        # Extract the heading (could be multiple lines if it's a header with subtext)
        in_heading = True
        for line in lines:
            if in_heading:
                heading_lines.append(line)
                if not line.strip():  # Empty line marks the end of the heading section
                    in_heading = False
            else:
                content_lines.append(line)

        # If there are no content lines, just return the heading
        if not content_lines:
            return [join_lines(heading_lines)]

        # Otherwise, handle the content separately
        content_text = join_lines(content_lines)
        content_chunks = ParagraphHandler().split_if_needed(content_text, max_len)

        # Combine heading with first content chunk
        heading_text = join_lines(heading_lines)
        if len(heading_text) + len(content_chunks[0]) + 1 <= max_len:  # +1 for newline
            result = [heading_text + "\n" + content_chunks[0]]
            result.extend(content_chunks[1:])
        else:
            result = [heading_text]
            result.extend(content_chunks)

        return result


class TableHandler(ContentHandler):
    """Handler for tables."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        if len(lines) < 2:
            return False

        # Check if the content starts with a table row and has a separator line
        return is_table_row(lines[0]) and any(
            is_table_separator(line) for line in lines[:3]
        )

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Split a table if needed, ensuring headers are included in each part.
        """
        if len(content) <= max_len:
            return [content]

        lines = split_into_lines(content)

        # Find the header rows (usually the first row and the separator row)
        header_end_idx = 0
        for i, line in enumerate(lines):
            if is_table_separator(line):
                header_end_idx = i
                break

        if (
            header_end_idx == 0
        ):  # No separator found, try to find another way to identify headers
            if len(lines) >= 2 and is_table_row(lines[0]) and is_table_row(lines[1]):
                header_end_idx = 1

        # Include up to the row after the separator
        header_rows = lines[: header_end_idx + 1]
        data_rows = lines[header_end_idx + 1 :]

        chunks = []
        current_chunk = header_rows[:]
        current_length = len(join_lines(current_chunk))

        for row in data_rows:
            row_length = len(row) + 1  # +1 for newline
            if current_length + row_length > max_len and len(current_chunk) > len(
                header_rows
            ):
                chunks.append(join_lines(current_chunk))
                current_chunk = header_rows[:]
                current_length = len(join_lines(current_chunk))

            current_chunk.append(row)
            current_length += row_length

        if current_chunk:
            chunks.append(join_lines(current_chunk))

        return chunks


class CodeBlockHandler(ContentHandler):
    """Handler for code blocks."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        return lines and is_code_block_delimiter(lines[0])

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Code blocks should ideally not be split. If they must be due to hard_max_len,
        we'll ensure each part has proper code block delimiters.
        """
        if len(content) <= max_len:
            return [content]

        lines = split_into_lines(content)

        # Extract the opening delimiter with any language specification
        opening_delimiter = lines[0]
        language_spec = ""
        if opening_delimiter.strip() != "```":
            language_spec = opening_delimiter.strip()[3:]

        # Find the closing delimiter
        closing_idx = -1
        for i, line in enumerate(lines[1:], 1):
            if is_code_block_delimiter(line):
                closing_idx = i
                break

        if closing_idx == -1:  # No closing delimiter found
            return [content]

        closing_delimiter = lines[closing_idx]
        code_lines = lines[1:closing_idx]

        # If the code itself is too large, we need to split it
        code_text = join_lines(code_lines)
        chunks = []

        # Split code into chunks, respecting max_len
        current_pos = 0
        while current_pos < len(code_text):
            # Calculate how much code can fit in this chunk
            # taking into account the delimiters
            delimiters_length = (
                len(opening_delimiter) + len(closing_delimiter) + 2
            )  # +2 for newlines
            available_space = max_len - delimiters_length

            # Find a good split point
            end_pos = min(current_pos + available_space, len(code_text))

            # Try to split at a newline if possible
            if end_pos < len(code_text):
                last_newline = code_text.rfind("\n", current_pos, end_pos)
                if last_newline != -1:
                    end_pos = last_newline + 1

            # Extract this chunk of code
            chunk_code = code_text[current_pos:end_pos]

            # Create the complete chunk with delimiters
            chunk = f"{opening_delimiter}\n{chunk_code}\n{closing_delimiter}"
            chunks.append(chunk)

            current_pos = end_pos

        return chunks


class ListHandler(ContentHandler):
    """Handler for ordered and unordered lists."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        return lines and is_list_item(lines[0])

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Split a list if needed, trying to keep list items intact.
        """
        if len(content) <= max_len:
            return [content]

        lines = split_into_lines(content)

        # Find list items with their indentation level
        list_items = []
        current_item_lines = []
        current_indent = None

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if is_list_item(line):
                # If we were building a previous item, add it
                if current_item_lines:
                    list_items.append((current_indent, current_item_lines))

                current_item_lines = [line]
                current_indent = indent
            elif stripped and (current_indent is None or indent > current_indent):
                # This is a continuation of the current item
                current_item_lines.append(line)
            elif not stripped:
                # Empty line, could be between list items
                if current_item_lines:
                    current_item_lines.append(line)
            else:
                # This is a new item or not part of the list
                if current_item_lines:
                    list_items.append((current_indent, current_item_lines))
                current_item_lines = [line]
                current_indent = indent

        # Add the last item if there is one
        if current_item_lines:
            list_items.append((current_indent, current_item_lines))

        # Split the list into chunks
        chunks = []
        current_chunk_items = []
        current_length = 0

        for indent, item_lines in list_items:
            item_text = join_lines(item_lines)
            item_length = len(item_text) + 1  # +1 for newline

            if current_length + item_length > max_len and current_chunk_items:
                chunks.append(
                    join_lines(
                        [line for _, lines in current_chunk_items for line in lines]
                    )
                )
                current_chunk_items = []
                current_length = 0

            current_chunk_items.append((indent, item_lines))
            current_length += item_length

        if current_chunk_items:
            chunks.append(
                join_lines([line for _, lines in current_chunk_items for line in lines])
            )

        return chunks


class BlockquoteHandler(ContentHandler):
    """Handler for blockquotes."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        return lines and is_blockquote(lines[0])

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Split a blockquote if needed, ensuring each part is a proper blockquote.
        """
        if len(content) <= max_len:
            return [content]

        lines = split_into_lines(content)

        # Collect blockquote lines and non-blockquote lines
        blockquote_lines = []

        for line in lines:
            if is_blockquote(line) or not line.strip():
                blockquote_lines.append(line)
            else:
                # This is not part of the blockquote
                break

        # If the entire content is a blockquote, we need to split it
        if len(blockquote_lines) == len(lines):
            chunks = []
            current_chunk_lines = []
            current_length = 0

            for line in blockquote_lines:
                line_length = len(line) + 1  # +1 for newline

                if current_length + line_length > max_len and current_chunk_lines:
                    chunks.append(join_lines(current_chunk_lines))
                    current_chunk_lines = []
                    current_length = 0

                current_chunk_lines.append(line)
                current_length += line_length

            if current_chunk_lines:
                chunks.append(join_lines(current_chunk_lines))

            return chunks
        else:
            # Handle the blockquote and non-blockquote parts separately
            blockquote_text = join_lines(blockquote_lines)
            remaining_text = join_lines(lines[len(blockquote_lines) :])

            blockquote_chunks = self.split_if_needed(blockquote_text, max_len)
            remaining_chunks = ParagraphHandler().split_if_needed(
                remaining_text, max_len
            )

            return blockquote_chunks + remaining_chunks


class ParagraphHandler(ContentHandler):
    """Handler for regular paragraphs."""

    def can_handle(self, content: str) -> bool:
        # This is a fallback handler for regular text
        return True

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Split a paragraph if needed, trying to split at sentence boundaries or whitespace.
        """
        if len(content) <= max_len:
            return [content]

        chunks = []
        remaining = content

        while len(remaining) > max_len:
            # Try to find a good split point
            split_point = None

            # Try to split at paragraph boundary
            match = re.search(r"\n\s*\n", remaining[:max_len])
            if match:
                split_point = match.end()
            else:
                # Try to split at sentence boundary
                sentence_end = re.finditer(r"[.!?]\s+", remaining[:max_len])
                last_sentence_end = None
                for match in sentence_end:
                    last_sentence_end = match.end()

                if last_sentence_end:
                    split_point = last_sentence_end
                else:
                    # Try to split at line break
                    line_break = remaining[:max_len].rfind("\n")
                    if line_break != -1:
                        split_point = line_break + 1
                    else:
                        # Last resort: split at word boundary
                        word_boundary = re.finditer(r"\s+", remaining[:max_len])
                        last_word_boundary = None
                        for match in word_boundary:
                            last_word_boundary = match.end()

                        if last_word_boundary:
                            split_point = last_word_boundary
                        else:
                            # If no good split point found, just split at max_len
                            split_point = max_len

            chunks.append(remaining[:split_point])
            remaining = remaining[split_point:]

        if remaining:
            chunks.append(remaining)

        return chunks


class FootnoteHandler(ContentHandler):
    """Handler for footnotes."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        return any(
            is_footnote_def(line) or contains_footnote_ref(line) for line in lines
        )

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Split content with footnotes, trying to keep footnote references with their definitions.
        """
        if len(content) <= max_len:
            return [content]

        # Extract footnote definitions
        lines = split_into_lines(content)
        footnote_defs = {}
        footnote_def_lines = set()

        for i, line in enumerate(lines):
            if is_footnote_def(line):
                match = re.match(r"^\[\^([^\]]+)\]:\s+.*$", line.strip())
                if match:
                    footnote_id = match.group(1)
                    footnote_defs[footnote_id] = i
                    footnote_def_lines.add(i)

        # Extract footnote references
        footnote_refs = {}
        for i, line in enumerate(lines):
            for match in re.finditer(r"\[\^([^\]]+)\]", line):
                footnote_id = match.group(1)
                if footnote_id not in footnote_refs:
                    footnote_refs[footnote_id] = []
                footnote_refs[footnote_id].append(i)

        # Try to split the content while keeping references with definitions
        chunks = []
        current_chunk_lines = []
        current_length = 0
        current_refs = set()

        for i, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for newline

            # Check if this line has footnote references
            for ref_id, ref_lines in footnote_refs.items():
                if i in ref_lines:
                    current_refs.add(ref_id)

            # If adding this line would exceed max_len, start a new chunk
            if current_length + line_length > max_len and current_chunk_lines:
                # Before finalizing the chunk, add any footnote definitions that are referenced
                for ref_id in current_refs:
                    if (
                        ref_id in footnote_defs
                        and footnote_defs[ref_id] not in footnote_def_lines
                    ):
                        def_line = lines[footnote_defs[ref_id]]
                        if current_length + len(def_line) + 1 <= max_len:
                            current_chunk_lines.append(def_line)
                            current_length += len(def_line) + 1
                            footnote_def_lines.add(footnote_defs[ref_id])

                chunks.append(join_lines(current_chunk_lines))
                current_chunk_lines = []
                current_length = 0
                current_refs = set()

            current_chunk_lines.append(line)
            current_length += line_length

        # Add the last chunk with any remaining footnote definitions
        if current_chunk_lines:
            for ref_id in current_refs:
                if (
                    ref_id in footnote_defs
                    and footnote_defs[ref_id] not in footnote_def_lines
                ):
                    def_line = lines[footnote_defs[ref_id]]
                    if current_length + len(def_line) + 1 <= max_len:
                        current_chunk_lines.append(def_line)
                        current_length += len(def_line) + 1
                        footnote_def_lines.add(footnote_defs[ref_id])

            chunks.append(join_lines(current_chunk_lines))

        return chunks


class HtmlHandler(ContentHandler):
    """Handler for embedded HTML."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        return any(is_html_tag(line) for line in lines)

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        Split HTML content, trying to keep HTML tags intact.
        """
        if len(content) <= max_len:
            return [content]

        # Try to identify HTML blocks with their opening and closing tags
        lines = split_into_lines(content)
        html_blocks = []
        current_block = []
        open_tags = []
        in_html_block = False

        for line in lines:
            # Check for opening and closing tags
            opening_tags = re.findall(r"<([^/][^>]*?)[\s>]", line)
            closing_tags = re.findall(r"</([^>]+)>", line)

            if (
                not in_html_block
                and opening_tags
                and not all(tag in closing_tags for tag in opening_tags)
            ):
                in_html_block = True
                open_tags.extend(
                    t.split()[0]
                    for t in opening_tags
                    if t.split()[0] not in closing_tags
                )

            if in_html_block:
                current_block.append(line)

                for tag in closing_tags:
                    if tag in open_tags:
                        open_tags.remove(tag)

                if not open_tags:
                    in_html_block = False
                    html_blocks.append(current_block)
                    current_block = []
            else:
                if current_block:
                    current_block.append(line)
                else:
                    html_blocks.append([line])

        if current_block:
            html_blocks.append(current_block)

        # Split the content while trying to keep HTML blocks intact
        chunks = []
        current_chunk_lines = []
        current_length = 0

        for block in html_blocks:
            block_text = join_lines(block)
            block_length = len(block_text) + 1  # +1 for newline

            if current_length + block_length > max_len and current_chunk_lines:
                chunks.append(join_lines(current_chunk_lines))
                current_chunk_lines = []
                current_length = 0

            if block_length > max_len:
                # If the block itself is too large, we need to split it
                # but this might break HTML structure
                block_chunks = ParagraphHandler().split_if_needed(block_text, max_len)

                if current_chunk_lines:
                    chunks.append(join_lines(current_chunk_lines))
                    current_chunk_lines = []
                    current_length = 0

                chunks.extend(block_chunks)
            else:
                current_chunk_lines.extend(block)
                current_length += block_length

        if current_chunk_lines:
            chunks.append(join_lines(current_chunk_lines))

        return chunks


class YamlFrontMatterHandler(ContentHandler):
    """Handler for YAML front matter."""

    def can_handle(self, content: str) -> bool:
        lines = split_into_lines(content)
        if len(lines) < 2:
            return False

        return lines[0].strip() == "---" and "---" in [
            line.strip() for line in lines[1:]
        ]

    def split_if_needed(self, content: str, max_len: int) -> List[str]:
        """
        YAML front matter should be kept in a single chunk.
        """
        if len(content) <= max_len:
            return [content]

        lines = split_into_lines(content)

        # Find the end of front matter
        front_matter_end = 0
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                front_matter_end = i
                break

        front_matter = lines[: front_matter_end + 1]
        remaining = lines[front_matter_end + 1 :]

        # Keep front matter in one chunk
        front_matter_text = join_lines(front_matter)

        # Handle remaining content
        if remaining:
            remaining_text = join_lines(remaining)
            remaining_chunks = ParagraphHandler().split_if_needed(
                remaining_text, max_len
            )

            # Try to combine front matter with first chunk of remaining content
            if len(front_matter_text) + len(remaining_chunks[0]) + 1 <= max_len:
                result = [front_matter_text + "\n" + remaining_chunks[0]]
                result.extend(remaining_chunks[1:])
            else:
                result = [front_matter_text]
                result.extend(remaining_chunks)

            return result
        else:
            return [front_matter_text]
