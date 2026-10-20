"""
Main Markdown chunking strategy class.
"""

import re
import yaml
from collections import defaultdict
import hashlib
from typing import List, Tuple, Dict, Set, Optional
import os.path

from .utils import (
    split_into_lines,
    join_lines,
    compute_hash,
    is_heading,
    is_table_row,
    is_table_separator,
    is_code_block_delimiter,
    is_list_item,
    is_blockquote,
    is_horizontal_rule,
    is_yaml_delimiter,
    is_html_tag,
    is_footnote_def,
    get_heading_level,
)
from .content_handlers import (
    ContentHandler,
    HeadingHandler,
    TableHandler,
    CodeBlockHandler,
    ListHandler,
    BlockquoteHandler,
    ParagraphHandler,
    FootnoteHandler,
    HtmlHandler,
    YamlFrontMatterHandler,
)


class MarkdownChunkingStrategy:
    """
    A strategy for chunking markdown documents that preserves structural elements,
    manages chunk sizes, prevents duplicates, and handles headers and footers.
    """

    def __init__(
        self,
        min_chunk_len: int = 512,
        soft_max_len: int = 1024,
        hard_max_len: int = 2048,
        detect_headers_footers: bool = True,
        remove_duplicates: bool = True,
        heading_based_chunking: bool = True,
        min_content_after_heading: int = 200,
        similarity_threshold: float = 0.85,
        parallel_processing: bool = False,
        max_workers: int = None,
        add_metadata: bool = False,
        document_title: str = None,
        source_document: str = None,
    ):
        """
        Initialize the chunking strategy with configurable parameters.

        Args:
            min_chunk_len: Minimum length of a chunk in characters
            soft_max_len: Preferred maximum length of a chunk
            hard_max_len: Absolute maximum length of a chunk
            detect_headers_footers: Whether to detect and remove headers and footers
            remove_duplicates: Whether to remove duplicate chunks
            heading_based_chunking: Whether to use headings as primary chunking points
            min_content_after_heading: Minimum content required after a heading to consider it "substantial"
            similarity_threshold: Threshold for fuzzy duplicate detection (0.0-1.0)
            parallel_processing: Whether to use parallel processing for large documents
            max_workers: Maximum number of workers for parallel processing (None = auto)
            add_metadata: Whether to embed metadata in each chunk as YAML front matter
            document_title: Title of the document (extracted from file or provided)
            source_document: Path or identifier of the source document
        """
        self.min_chunk_len = min_chunk_len
        self.soft_max_len = soft_max_len
        self.hard_max_len = hard_max_len
        self.detect_headers_footers = detect_headers_footers
        self.remove_duplicates = remove_duplicates
        self.heading_based_chunking = heading_based_chunking
        self.min_content_after_heading = min_content_after_heading
        self.similarity_threshold = similarity_threshold
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        self.add_metadata = add_metadata
        self.document_title = document_title
        self.source_document = source_document

        # Initialize content handlers
        self.content_handlers = [
            YamlFrontMatterHandler(),
            HeadingHandler(),
            TableHandler(),
            CodeBlockHandler(),
            ListHandler(),
            BlockquoteHandler(),
            FootnoteHandler(),
            HtmlHandler(),
            ParagraphHandler(),  # This is the fallback handler
        ]

    def chunk_markdown(self, markdown_text: str) -> List[str]:
        """
        Main method to chunk the markdown text according to the rules.

        Args:
            markdown_text: The markdown text to chunk

        Returns:
            A list of markdown chunks
        """
        # Extract document title if not provided
        if self.document_title is None:
            self.document_title = self._extract_document_title(markdown_text)

        # Store original content for validation
        original_content = markdown_text

        # 1. Detect and remove headers and footers
        removed_headers = None
        removed_footers = None
        if self.detect_headers_footers:
            # Store detected headers and footers before removing them
            lines = split_into_lines(markdown_text)
            removed_headers = self._find_repeating_header_pattern(lines)
            removed_footers = self._find_repeating_footer_pattern(lines)
            markdown_text = self._detect_headers_footers(markdown_text)

        # 2. Process with parallel execution if enabled for large documents
        if self.parallel_processing and len(markdown_text) > self.hard_max_len * 10:
            chunks = self._parallel_chunk_markdown(markdown_text)
        else:
            # 3. Split the markdown into logical blocks
            blocks = self._split_into_blocks(markdown_text)

            # 4. Create initial chunks based on heading structure if enabled
            if self.heading_based_chunking:
                chunks = self._create_heading_based_chunks(blocks)
            else:
                # Otherwise use the size-based chunking
                chunks = self._create_initial_chunks(blocks)

            # 5. Merge small chunks
            chunks = self._merge_small_chunks(chunks)

            # 6. Handle large chunks that exceed hard_max_len
            chunks = self._handle_large_chunks(chunks)

            # 7. Ensure no chunk ends with a heading or has insufficient content after a heading
            if self.heading_based_chunking:
                chunks = self._optimize_heading_chunks(chunks)

            # 8. Remove duplicate chunks if enabled
            if self.remove_duplicates:
                chunks = self._remove_duplicates(chunks)

                # 9. Detect and handle fuzzy duplicates if enabled
                if self.similarity_threshold < 1.0:
                    chunks = self._handle_fuzzy_duplicates(chunks)

        # 10. Generate enhanced metadata for the chunks
        metadata = self._generate_chunk_metadata(chunks)

        # 11. Enhance cross-chunk references
        chunks = self._enhance_cross_chunk_references(chunks, metadata)

        # 12. Embed metadata in chunks if enabled
        if self.add_metadata:
            chunks = self._add_metadata_to_chunks(chunks, metadata)

        # 13. Validate content completeness
        self._validate_content_completeness(
            original_content, chunks, removed_headers, removed_footers
        )

        return chunks

    def _validate_content_completeness(
        self,
        original_content: str,
        chunks: List[str],
        removed_headers: Optional[List[str]],
        removed_footers: Optional[List[str]],
    ) -> None:
        """
        Validate that no content is lost during chunking except for intentionally removed headers and footers.
        This method logs warnings if content appears to be lost.

        Args:
            original_content: The original markdown text before chunking
            chunks: The resulting chunks after processing
            removed_headers: The headers that were detected and potentially removed
            removed_footers: The footers that were detected and potentially removed
        """
        # Skip validation if chunking failed or produced empty results
        if not chunks:
            return

        # Join all chunks and normalize whitespace
        joined_chunks = "\n\n".join(chunks)

        # If headers/footers were removed, we need to account for that difference
        expected_content_diff = 0

        if removed_headers or removed_footers:
            # Calculate expected content reduction due to header/footer removal
            # Assuming headers/footers might be repeated ~3 times on average
            header_size = sum(len(line) for line in (removed_headers or [])) * 3
            footer_size = sum(len(line) for line in (removed_footers or [])) * 3
            expected_content_diff = header_size + footer_size

        # Compare content lengths accounting for intentional removals and some whitespace variance
        # Allow for small differences due to whitespace normalization
        whitespace_variance = (
            len(original_content) * 0.05
        )  # 5% allowance for whitespace
        actual_diff = abs(len(original_content) - len(joined_chunks))

        # Check if the difference exceeds our expectations
        if actual_diff > expected_content_diff + whitespace_variance:
            # This is a case where we may have lost content
            missing_bytes = actual_diff - expected_content_diff
            missing_percent = (missing_bytes / len(original_content)) * 100

            # Log a warning about potential content loss
            print(
                f"WARNING: Possible content loss during chunking! "
                f"Missing approximately {missing_bytes} bytes "
                f"({missing_percent:.2f}% of original content)."
            )

        # Additional check: make sure all major content elements are preserved
        # This does a rough content fingerprinting using headers, lists, and code blocks
        original_lines = split_into_lines(original_content)
        chunk_lines = split_into_lines(joined_chunks)

        # Check preservation of headings, not counting repeating headers that were removed
        original_headings = [
            line.strip()
            for line in original_lines
            if is_heading(line) and (not removed_headers or line not in removed_headers)
        ]
        chunk_headings = [line.strip() for line in chunk_lines if is_heading(line)]

        # Get counts of important elements
        original_tables = sum(1 for line in original_lines if is_table_row(line))
        chunk_tables = sum(1 for line in chunk_lines if is_table_row(line))

        original_code_blocks = (
            sum(1 for line in original_lines if is_code_block_delimiter(line)) // 2
        )
        chunk_code_blocks = (
            sum(1 for line in chunk_lines if is_code_block_delimiter(line)) // 2
        )

        original_list_items = sum(1 for line in original_lines if is_list_item(line))
        chunk_list_items = sum(1 for line in chunk_lines if is_list_item(line))

        # Check if counts match allowing for some small variations
        headings_match = len(set(original_headings)) == len(set(chunk_headings))
        tables_match = abs(original_tables - chunk_tables) <= 2  # Allow small variance
        code_blocks_match = original_code_blocks == chunk_code_blocks
        list_items_match = abs(original_list_items - chunk_list_items) <= 2

        # Log any discrepancies
        if not headings_match:
            print(
                f"WARNING: Heading count mismatch! Original: {len(set(original_headings))}, "
                f"Chunks: {len(set(chunk_headings))}"
            )

        if not tables_match:
            print(
                f"WARNING: Table row count mismatch! Original: {original_tables}, "
                f"Chunks: {chunk_tables}"
            )

        if not code_blocks_match:
            print(
                f"WARNING: Code block count mismatch! Original: {original_code_blocks}, "
                f"Chunks: {chunk_code_blocks}"
            )

        if not list_items_match:
            print(
                f"WARNING: List item count mismatch! Original: {original_list_items}, "
                f"Chunks: {chunk_list_items}"
            )

    def _parallel_chunk_markdown(self, markdown_text: str) -> List[str]:
        """
        Process large documents in parallel using ProcessPoolExecutor.

        Args:
            markdown_text: The markdown text to chunk

        Returns:
            A list of markdown chunks
        """
        try:
            from concurrent.futures import ProcessPoolExecutor
            import os

            # Determine optimal number of workers
            if self.max_workers is None:
                # Use CPU count - 1 (leave one core free), but minimum of 1
                self.max_workers = max(1, os.cpu_count() - 1)

            # First, split the document into major sections (e.g., at top-level headings)
            sections = self._split_into_major_sections(markdown_text)

            # Process each section in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a non-parallel version of the chunking strategy for sections
                section_chunker = MarkdownChunkingStrategy(
                    min_chunk_len=self.min_chunk_len,
                    soft_max_len=self.soft_max_len,
                    hard_max_len=self.hard_max_len,
                    detect_headers_footers=False,  # Already done on the whole document
                    remove_duplicates=False,  # Will do this after combining
                    heading_based_chunking=self.heading_based_chunking,
                    min_content_after_heading=self.min_content_after_heading,
                    similarity_threshold=1.0,  # Skip fuzzy duplicate detection for sections
                    parallel_processing=False,  # No nested parallelism
                )

                # Submit tasks to process each section
                futures = []
                for section in sections:
                    future = executor.submit(section_chunker.chunk_markdown, section)
                    futures.append(future)

                # Collect results
                all_chunks = []
                for future in futures:
                    all_chunks.extend(future.result())

            # Perform post-processing on the combined chunks
            if self.remove_duplicates:
                all_chunks = self._remove_duplicates(all_chunks)

                if self.similarity_threshold < 1.0:
                    all_chunks = self._handle_fuzzy_duplicates(all_chunks)

            return all_chunks

        except ImportError:
            # Fall back to non-parallel processing if concurrent.futures is not available
            return self._non_parallel_chunk_markdown(markdown_text)

    def _non_parallel_chunk_markdown(self, markdown_text: str) -> List[str]:
        """
        Fallback to non-parallel processing.

        Args:
            markdown_text: The markdown text to chunk

        Returns:
            A list of markdown chunks
        """
        # Instead of calling chunk_markdown recursively, implement the non-parallel version directly
        # Split the markdown into logical blocks
        blocks = self._split_into_blocks(markdown_text)

        # Create initial chunks based on heading structure if enabled
        if self.heading_based_chunking:
            chunks = self._create_heading_based_chunks(blocks)
        else:
            # Otherwise use the size-based chunking
            chunks = self._create_initial_chunks(blocks)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        # Handle large chunks that exceed hard_max_len
        chunks = self._handle_large_chunks(chunks)

        # Ensure no chunk ends with a heading or has insufficient content after a heading
        if self.heading_based_chunking:
            chunks = self._optimize_heading_chunks(chunks)

        # Remove duplicate chunks if enabled
        if self.remove_duplicates:
            chunks = self._remove_duplicates(chunks)

            # Detect and handle fuzzy duplicates if enabled
            if self.similarity_threshold < 1.0:
                chunks = self._handle_fuzzy_duplicates(chunks)

        return chunks

    def _split_into_major_sections(self, markdown_text: str) -> List[str]:
        """
        Split the document into major sections for parallel processing.

        Args:
            markdown_text: The markdown text to split

        Returns:
            A list of sections
        """
        lines = split_into_lines(markdown_text)
        sections = []
        current_section = []

        for i, line in enumerate(lines):
            current_section.append(line)

            # Split on top-level headings (# Heading) if not at the start
            if (
                is_heading(line)
                and get_heading_level(line) == 1
                and i > 0
                and i < len(lines) - 1
            ):
                sections.append(join_lines(current_section))
                current_section = [line]

        # Add the final section if there's content
        if current_section:
            sections.append(join_lines(current_section))

        # If no sections were created, return the whole document as one section
        if not sections:
            sections.append(markdown_text)

        return sections

    def _detect_headers_footers(self, markdown_text: str) -> str:
        """
        Detect and remove repeating headers and footers from the markdown text.

        Args:
            markdown_text: The markdown text to process

        Returns:
            The markdown text with headers and footers removed
        """
        lines = split_into_lines(markdown_text)
        if len(lines) < 10:  # Not enough lines to detect patterns
            return markdown_text

        # Look for repeating patterns at the beginning of sections
        header_pattern = self._find_repeating_header_pattern(lines)

        # Look for repeating patterns at the end of sections
        footer_pattern = self._find_repeating_footer_pattern(lines)

        # Remove the detected headers and footers
        if header_pattern:
            lines = self._remove_headers(lines, header_pattern)

        if footer_pattern:
            lines = self._remove_footers(lines, footer_pattern)

        return join_lines(lines)

    def _find_repeating_header_pattern(self, lines: List[str]) -> Optional[List[str]]:
        """
        Find a repeating header pattern at the beginning of sections using advanced pattern recognition.

        Args:
            lines: The lines of the markdown text

        Returns:
            A list of lines representing the header pattern, or None if not found
        """
        # Enhanced algorithm for header detection
        # Look for section breaks (empty lines, horizontal rules, or headings)
        section_breaks = []
        for i, line in enumerate(lines):
            if (
                not line.strip()
                or is_horizontal_rule(line)
                or (is_heading(line) and get_heading_level(line) <= 2)
            ):  # Top-level or second-level headings
                section_breaks.append(i)

        if len(section_breaks) < 3:  # Not enough sections to detect patterns
            return None

        # Get the first few lines of each section
        max_header_lines = 7  # Consider up to 7 lines for header (extended from 5)
        section_starts = []

        for i in range(len(section_breaks) - 1):
            start_idx = section_breaks[i]
            end_idx = min(section_breaks[i] + max_header_lines, section_breaks[i + 1])
            section = lines[start_idx:end_idx]
            # Remove empty lines at the beginning
            while section and not section[0].strip():
                section = section[1:]

            if section:
                section_starts.append(section)

        # Compare the starts of each section to find patterns
        if len(section_starts) < 3:  # Not enough sections with content
            return None

        # Check for exact matches with enhanced pattern recognition
        for length in range(1, max_header_lines + 1):
            potential_headers = defaultdict(int)

            for section in section_starts:
                if len(section) >= length:
                    # Create a normalized representation of the header for comparison
                    # This can help detect headers with slight variations
                    normalized = [
                        self._normalize_line(line) for line in section[:length]
                    ]
                    header = tuple(normalized)
                    potential_headers[header] += 1

            # A header must appear in at least 3 sections with good coverage
            for header, count in potential_headers.items():
                # Require at least 50% coverage (header appears in at least half of sections)
                if count >= 3 and len(section_starts) / count <= 2:
                    # Convert back to the original lines, not normalized
                    original_lines = []
                    for section in section_starts:
                        if len(section) >= length:
                            normalized = tuple(
                                self._normalize_line(line) for line in section[:length]
                            )
                            if normalized == header:
                                original_lines = section[:length]
                                break

                    return original_lines

        return None

    def _normalize_line(self, line: str) -> str:
        """
        Normalize a line for pattern matching to detect slight variations.
        This helps with fuzzy header/footer matching.

        Args:
            line: The line to normalize

        Returns:
            A normalized version of the line
        """
        # Remove extra whitespace
        normalized = " ".join(line.strip().split())

        # Remove or normalize page numbers and dates
        normalized = re.sub(
            r"\b(page|p\.?)\s*\d+\b", "PAGE_NUM", normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "DATE", normalized)

        # Remove varying numbers
        normalized = re.sub(r"\b\d+\b", "NUM", normalized)

        return normalized

    def _find_repeating_footer_pattern(self, lines: List[str]) -> Optional[List[str]]:
        """
        Find a repeating footer pattern at the end of sections using advanced pattern recognition.

        Args:
            lines: The lines of the markdown text

        Returns:
            A list of lines representing the footer pattern, or None if not found
        """
        # Similar to header detection but with enhanced algorithm
        section_breaks = []
        for i, line in enumerate(lines):
            if (
                not line.strip()
                or is_horizontal_rule(line)
                or (is_heading(line) and get_heading_level(line) <= 2)
            ):
                section_breaks.append(i)

        if len(section_breaks) < 3:
            return None

        max_footer_lines = 7  # Consider up to 7 lines for footer (extended from 5)
        section_ends = []

        for i in range(len(section_breaks) - 1):
            end_idx = section_breaks[i + 1]
            start_idx = max(section_breaks[i], end_idx - max_footer_lines)
            section = lines[start_idx:end_idx]
            # Remove empty lines at the end
            while section and not section[-1].strip():
                section = section[:-1]

            if section:
                section_ends.append(section)

        if len(section_ends) < 3:
            return None

        for length in range(1, max_footer_lines + 1):
            potential_footers = defaultdict(int)

            for section in section_ends:
                if len(section) >= length:
                    # Normalize the footer lines for comparison
                    normalized = [
                        self._normalize_line(line) for line in section[-length:]
                    ]
                    footer = tuple(normalized)
                    potential_footers[footer] += 1

            for footer, count in potential_footers.items():
                if count >= 3 and len(section_ends) / count <= 2:
                    # Convert back to the original lines, not normalized
                    original_lines = []
                    for section in section_ends:
                        if len(section) >= length:
                            normalized = tuple(
                                self._normalize_line(line) for line in section[-length:]
                            )
                            if normalized == footer:
                                original_lines = section[-length:]
                                break

                    return original_lines

        return None

    def _remove_headers(self, lines: List[str], header_pattern: List[str]) -> List[str]:
        """
        Remove repeating headers from the text with improved handling of variations.

        Args:
            lines: The lines of the markdown text
            header_pattern: The header pattern to remove

        Returns:
            The lines with headers removed
        """
        result = []
        i = 0
        header_len = len(header_pattern)
        headers_found = 0
        headers_removed = 0  # Track how many headers were removed for logging

        while i < len(lines):
            # Check if this is a header
            is_header = False
            if i + header_len <= len(lines):
                potential_header = lines[i : i + header_len]
                # Check for exact match or normalized match
                is_exact_match = all(
                    ph.strip() == hp.strip()
                    for ph, hp in zip(potential_header, header_pattern)
                )

                if is_exact_match:
                    is_header = True
                else:
                    # Try a more flexible matching for slight variations
                    potential_normalized = [
                        self._normalize_line(line) for line in potential_header
                    ]
                    pattern_normalized = [
                        self._normalize_line(line) for line in header_pattern
                    ]

                    if all(
                        pn == pn2
                        for pn, pn2 in zip(potential_normalized, pattern_normalized)
                    ):
                        is_header = True

            if is_header:
                headers_found += 1
                # Skip the header but keep the first occurrence
                if headers_found == 1 or (i + header_len == len(lines)):
                    result.extend(lines[i : i + header_len])
                    # If we kept the header, don't count it as removed
                    i += header_len
                else:
                    # Remove this instance of the header
                    i += header_len
                    headers_removed += 1
            else:
                result.append(lines[i])
                i += 1

        # Log how many headers were found and removed
        if headers_removed > 0:
            print(f"Removed {headers_removed} recurring headers (kept the first one)")

        return result

    def _remove_footers(self, lines: List[str], footer_pattern: List[str]) -> List[str]:
        """
        Remove repeating footers from the text with improved handling of variations.

        Args:
            lines: The lines of the markdown text
            footer_pattern: The footer pattern to remove

        Returns:
            The lines with footers removed
        """
        result = []
        i = 0
        footer_len = len(footer_pattern)
        footers_found = 0

        while i < len(lines):
            # Check if this is a footer
            is_footer = False
            if i + footer_len <= len(lines):
                potential_footer = lines[i : i + footer_len]

                # Check for exact match or normalized match
                is_exact_match = all(
                    pf.strip() == ff.strip()
                    for pf, ff in zip(potential_footer, footer_pattern)
                )

                if is_exact_match:
                    is_footer = True
                else:
                    # Try a more flexible matching for slight variations
                    potential_normalized = [
                        self._normalize_line(line) for line in potential_footer
                    ]
                    pattern_normalized = [
                        self._normalize_line(line) for line in footer_pattern
                    ]

                    if all(
                        pn == pn2
                        for pn, pn2 in zip(potential_normalized, pattern_normalized)
                    ):
                        is_footer = True

            if is_footer:
                footers_found += 1
                # Skip the footer but keep the last occurrence
                if i + footer_len == len(lines):
                    result.extend(lines[i : i + footer_len])
                i += footer_len
            else:
                result.append(lines[i])
                i += 1

        return result

    def _split_into_blocks(self, markdown_text: str) -> List[Tuple[str, str, int]]:
        """
        Split the markdown text into logical blocks based on content type.

        Args:
            markdown_text: The markdown text to split

        Returns:
            A list of (block_type, block_content, heading_level) tuples
            heading_level is 0 for non-heading blocks, and 1-6 for heading blocks
        """
        lines = split_into_lines(markdown_text)
        blocks = []
        current_lines = []
        current_type = None
        current_heading_level = 0

        # Parse state flags
        in_code_block = False
        in_yaml_front_matter = False

        for line in lines:
            line_type = self._get_line_type(line)
            heading_level = get_heading_level(line) if is_heading(line) else 0

            # Update parse state
            if line_type == "code_block_delimiter":
                in_code_block = not in_code_block
            elif line_type == "yaml_delimiter" and not in_code_block:
                in_yaml_front_matter = not in_yaml_front_matter

            # Determine if we should start a new block
            start_new_block = False

            if not current_lines:
                # First line of the document
                current_lines.append(line)
                current_type = line_type
                current_heading_level = heading_level
                continue
            elif in_code_block or in_yaml_front_matter:
                # Inside a code block or YAML front matter, keep adding to current block
                current_lines.append(line)
            elif line_type == "heading" and not in_code_block:
                # Headings start new blocks
                start_new_block = True
                current_heading_level = heading_level
            elif line_type == "table_row" and not in_code_block:
                if current_type not in ["table_row", "table_separator"]:
                    start_new_block = True
            elif line_type == "list_item" and not in_code_block:
                if current_type != "list_item" and current_type != "paragraph":
                    start_new_block = True
            elif line_type == "blockquote" and not in_code_block:
                if current_type != "blockquote":
                    start_new_block = True
            elif not line.strip() and current_type == "paragraph":
                # Empty line after paragraph: end the paragraph
                start_new_block = True
            elif (
                line.strip()
                and not line.strip().startswith(">")
                and current_type == "blockquote"
            ):
                # End of blockquote
                start_new_block = True
            elif line_type == "horizontal_rule":
                # Horizontal rules start new blocks
                start_new_block = True

            if start_new_block and current_lines:
                # Add the current block to the list
                block_content = join_lines(current_lines)
                blocks.append((current_type, block_content, current_heading_level))
                current_lines = [line]
                current_type = line_type
                current_heading_level = heading_level
            else:
                # Continue current block
                current_lines.append(line)

        # Add the last block
        if current_lines:
            block_content = join_lines(current_lines)
            blocks.append((current_type, block_content, current_heading_level))

        return blocks

    def _get_line_type(self, line: str) -> str:
        """
        Determine the type of a markdown line.

        Args:
            line: The line to check

        Returns:
            The type of the line as a string
        """
        if not line.strip():
            return "empty"
        elif is_heading(line):
            return "heading"
        elif is_table_row(line):
            return "table_row"
        elif is_table_separator(line):
            return "table_separator"
        elif is_code_block_delimiter(line):
            return "code_block_delimiter"
        elif is_list_item(line):
            return "list_item"
        elif is_blockquote(line):
            return "blockquote"
        elif is_horizontal_rule(line):
            return "horizontal_rule"
        elif is_yaml_delimiter(line):
            return "yaml_delimiter"
        elif is_html_tag(line):
            return "html"
        elif is_footnote_def(line):
            return "footnote_def"
        else:
            return "paragraph"

    def _create_heading_based_chunks(
        self, blocks: List[Tuple[str, str, int]]
    ) -> List[str]:
        """
        Create chunks based on heading structure, prioritizing keeping headings with their content.

        Args:
            blocks: List of (block_type, block_content, heading_level) tuples

        Returns:
            A list of markdown chunks
        """
        chunks = []
        current_chunk_blocks = []
        current_length = 0
        current_main_heading = None
        contains_non_heading = False

        for block_type, block_content, heading_level in blocks:
            block_length = len(block_content)

            # Check if this is a heading that should start a new chunk
            is_significant_heading = (
                block_type == "heading"
                and heading_level <= 2  # Consider levels 1-2 as major section breaks
                and current_length
                > 0  # Don't start a new chunk if we haven't added anything yet
            )

            # Start a new chunk if:
            # 1. This is a significant heading and we're using heading-based chunking
            # 2. Adding this block would exceed soft_max_len and it's not a heading followed by its first content
            should_start_new_chunk = (
                (
                    is_significant_heading
                    and current_length > self.min_chunk_len
                    and contains_non_heading  # Only start a new chunk if current one has non-heading content
                )
                or (
                    current_length + block_length > self.soft_max_len
                    and not (
                        current_main_heading is not None
                        and len(current_chunk_blocks) == 1
                    )
                )
            )

            if should_start_new_chunk and current_chunk_blocks:
                chunk_text = "\n\n".join(current_chunk_blocks)
                chunks.append(chunk_text)
                current_chunk_blocks = []
                current_length = 0
                current_main_heading = None
                contains_non_heading = False

            # Update current main heading if this is a heading
            if block_type == "heading":
                if heading_level <= 2 or current_main_heading is None:
                    current_main_heading = (heading_level, block_content)
            else:
                contains_non_heading = True

            current_chunk_blocks.append(block_content)
            current_length += block_length + 2  # +2 for the newlines between blocks

        # Add the last chunk
        if current_chunk_blocks:
            chunk_text = "\n\n".join(current_chunk_blocks)
            chunks.append(chunk_text)

        return chunks

    def _create_initial_chunks(self, blocks: List[Tuple[str, str, int]]) -> List[str]:
        """
        Create initial chunks from blocks based on soft_max_len.

        Args:
            blocks: List of (block_type, block_content, heading_level) tuples

        Returns:
            A list of markdown chunks
        """
        chunks = []
        current_chunk_blocks = []
        current_length = 0

        for block_type, block_content, _ in blocks:
            block_length = len(block_content)

            # Start a new chunk if this block would exceed soft_max_len
            if (
                current_length + block_length > self.soft_max_len
                and current_chunk_blocks
            ):
                chunk_text = "\n\n".join(current_chunk_blocks)
                chunks.append(chunk_text)
                current_chunk_blocks = []
                current_length = 0

            current_chunk_blocks.append(block_content)
            current_length += block_length + 2  # +2 for the newlines between blocks

        # Add the last chunk
        if current_chunk_blocks:
            chunk_text = "\n\n".join(current_chunk_blocks)
            chunks.append(chunk_text)

        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are smaller than min_chunk_len with adjacent chunks.

        Args:
            chunks: List of markdown chunks

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        result = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If this chunk is too small and not the last one, try to merge with the next
            if len(current) < self.min_chunk_len and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                merged = current + "\n\n" + next_chunk

                # If merged chunk is within hard_max_len, merge them
                if len(merged) <= self.hard_max_len:
                    result.append(merged)
                    i += 2
                else:
                    # If we can't merge, keep the small chunk
                    result.append(current)
                    i += 1
            else:
                result.append(current)
                i += 1

        return result

    def _optimize_heading_chunks(self, chunks: List[str]) -> List[str]:
        """
        Ensure no chunk ends with a heading or has insufficient content after a heading.

        Args:
            chunks: List of markdown chunks

        Returns:
            Optimized list of chunks
        """
        result = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            lines = split_into_lines(current_chunk)

            # Check if the chunk ends with a heading
            ends_with_heading = False
            heading_index = -1

            # Scan from the end to find if it ends with a heading
            for j in range(len(lines) - 1, -1, -1):
                if not lines[j].strip():
                    continue  # Skip empty lines

                if is_heading(lines[j]):
                    ends_with_heading = True
                    heading_index = j
                break

            # Check if there's insufficient content after a heading
            has_insufficient_content = False
            insufficient_content_index = -1

            for j in range(len(lines) - 1):
                if is_heading(lines[j]):
                    # Calculate content length after this heading
                    content_after = join_lines(lines[j + 1 :])
                    if 0 < len(content_after) < self.min_content_after_heading:
                        has_insufficient_content = True
                        insufficient_content_index = j
                        break

            # Check if the chunk contains only headings without actual content
            contains_only_headings = True
            has_tables = False
            has_code_blocks = False
            has_lists = False

            in_code_block = False
            for line in lines:
                stripped = line.strip()

                # Check for code blocks
                if is_code_block_delimiter(line):
                    in_code_block = not in_code_block
                    has_code_blocks = True
                    contains_only_headings = False
                    break

                # Check for tables
                if is_table_row(line) or is_table_separator(line):
                    has_tables = True
                    contains_only_headings = False
                    break

                # Check for lists
                if is_list_item(line):
                    has_lists = True
                    contains_only_headings = False
                    break

                # Check for regular content (neither heading nor empty)
                if stripped and not is_heading(line) and not stripped.isspace():
                    contains_only_headings = False
                    break

            # Handle the cases
            if contains_only_headings and i < len(chunks) - 1:
                # If the chunk contains only headings and there's a next chunk,
                # merge it with the next chunk
                next_chunk = chunks[i + 1]
                merged_chunk = current_chunk + "\n\n" + next_chunk

                # Only apply if the result doesn't exceed hard_max_len
                if len(merged_chunk) <= self.hard_max_len:
                    chunks[i + 1] = merged_chunk  # Update for next iteration
                    # Skip adding the current chunk to result as it's merged with next
                else:
                    result.append(current_chunk)  # Keep as is
            elif ends_with_heading and i < len(chunks) - 1:
                # If the chunk ends with a heading and there's a next chunk,
                # move the heading to the next chunk
                next_chunk = chunks[i + 1]
                new_current = join_lines(lines[:heading_index])
                new_next = join_lines(lines[heading_index:]) + "\n\n" + next_chunk

                # Only apply if the result doesn't exceed hard_max_len
                if len(new_next) <= self.hard_max_len:
                    result.append(new_current)
                    chunks[i + 1] = new_next  # Update for next iteration
                else:
                    result.append(current_chunk)  # Keep as is

            elif has_insufficient_content and i < len(chunks) - 1:
                # If there's insufficient content after a heading and there's a next chunk,
                # try to move content from the next chunk
                next_chunk = chunks[i + 1]
                next_lines = split_into_lines(next_chunk)

                # Find a good split point in the next chunk
                split_index = self._find_good_split_point(next_lines)

                if split_index > 0:
                    # Move content from the next chunk to this one
                    new_current = (
                        current_chunk + "\n\n" + join_lines(next_lines[:split_index])
                    )
                    new_next = join_lines(next_lines[split_index:])

                    # Only apply if within limits
                    if (
                        len(new_current) <= self.hard_max_len
                        and len(new_next) >= self.min_chunk_len
                    ):
                        result.append(new_current)
                        chunks[i + 1] = new_next  # Update for next iteration
                    else:
                        result.append(current_chunk)  # Keep as is
                else:
                    result.append(current_chunk)  # Keep as is
            else:
                result.append(current_chunk)  # Keep as is

            i += 1

        return result

    def _find_good_split_point(self, lines: List[str]) -> int:
        """
        Find a good point to split lines, preferably before a heading or after a paragraph.

        Args:
            lines: List of lines to analyze

        Returns:
            Index where the lines should be split
        """
        # First, look for a heading (best split point)
        for i, line in enumerate(lines):
            if is_heading(line):
                return i

        # Next, look for a paragraph break (empty line)
        for i, line in enumerate(lines):
            if not line.strip() and i > 0 and i < len(lines) - 1:
                return i + 1  # Split after the empty line

        # If no good split point, split at the middle
        return len(lines) // 2

    def _handle_large_chunks(self, chunks: List[str]) -> List[str]:
        """
        Handle chunks that exceed hard_max_len by splitting them at logical points.

        Args:
            chunks: List of markdown chunks

        Returns:
            List of chunks all within hard_max_len
        """
        result = []

        for chunk in chunks:
            if len(chunk) <= self.hard_max_len:
                result.append(chunk)
            else:
                # Find the appropriate content handler
                handler = self._find_handler_for_content(chunk)
                split_chunks = handler.split_if_needed(chunk, self.hard_max_len)
                result.extend(split_chunks)

        return result

    def _find_handler_for_content(self, content: str) -> ContentHandler:
        """
        Find the appropriate content handler for the given content.

        Args:
            content: The content to find a handler for

        Returns:
            A ContentHandler that can handle the content
        """
        for handler in self.content_handlers:
            if handler.can_handle(content):
                return handler

        # Default to paragraph handler
        return ParagraphHandler()

    def _remove_duplicates(self, chunks: List[str]) -> List[str]:
        """
        Remove duplicate chunks based on content hash.

        Args:
            chunks: List of markdown chunks

        Returns:
            List of unique chunks
        """
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_hash = compute_hash(chunk)
            if chunk_hash not in seen_hashes:
                unique_chunks.append(chunk)
                seen_hashes.add(chunk_hash)

        return unique_chunks

    def _handle_fuzzy_duplicates(self, chunks: List[str]) -> List[str]:
        """
        Identify and manage near-duplicate content using fuzzy matching.

        Args:
            chunks: List of markdown chunks

        Returns:
            List of chunks with fuzzy duplicates handled
        """
        try:
            from difflib import SequenceMatcher

            # If no chunks or only one chunk, nothing to do
            if len(chunks) <= 1:
                return chunks

            # Compare each pair of chunks for similarity
            keep_indexes = set(range(len(chunks)))  # Start with keeping all chunks
            similarity_groups = {}  # Groups of similar chunks

            for i in range(len(chunks)):
                if i not in keep_indexes:
                    continue  # Skip chunks already marked for removal

                for j in range(i + 1, len(chunks)):
                    if j not in keep_indexes:
                        continue  # Skip chunks already marked for removal

                    # Calculate similarity using SequenceMatcher
                    matcher = SequenceMatcher(None, chunks[i], chunks[j])
                    similarity = matcher.ratio()

                    if similarity >= self.similarity_threshold:
                        # Group similar chunks
                        if i not in similarity_groups:
                            similarity_groups[i] = [i]

                        similarity_groups[i].append(j)
                        keep_indexes.discard(j)  # Mark j for removal

            # Keep the most comprehensive chunk from each similarity group
            result = []

            for i in range(len(chunks)):
                if i in keep_indexes:
                    if i in similarity_groups:
                        # Find the longest chunk in the group (most comprehensive)
                        group = similarity_groups[i]
                        best_chunk = max([chunks[idx] for idx in group], key=len)
                        result.append(best_chunk)
                    else:
                        result.append(chunks[i])

            return result

        except ImportError:
            # If difflib is not available, just return the original chunks
            return chunks

    def _generate_chunk_metadata(self, chunks: List[str]) -> dict:
        """
        Generate comprehensive metadata for each chunk including heading structure and relationships.

        Args:
            chunks: List of markdown chunks

        Returns:
            Dictionary containing metadata for all chunks
        """
        metadata = {"total_chunks": len(chunks), "chunks": []}

        # Extract headings from each chunk to build a hierarchy
        all_headings = []  # List of (chunk_idx, heading_level, heading_text)

        for chunk_idx, chunk in enumerate(chunks):
            lines = split_into_lines(chunk)
            chunk_headings = []

            for i, line in enumerate(lines):
                if is_heading(line):
                    level = get_heading_level(line)
                    text = line.strip("#").strip()
                    chunk_headings.append((level, text, i))

            # Store the headings for this chunk
            all_headings.append((chunk_idx, chunk_headings))

            # Create basic metadata for the chunk
            chunk_metadata = {
                "id": chunk_idx + 1,
                "length": len(chunk),
                "headings": [
                    {"level": level, "text": text} for level, text, _ in chunk_headings
                ],
                "first_heading": chunk_headings[0][1] if chunk_headings else None,
                "content_types": self._identify_content_types(chunk),
            }

            metadata["chunks"].append(chunk_metadata)

        # Build relationships between chunks based on heading hierarchy
        for i, (chunk_idx, chunk_headings) in enumerate(all_headings):
            # Look for parent/child relationships
            if i > 0 and chunk_headings:
                prev_chunk_idx, prev_chunk_headings = all_headings[i - 1]

                # See if this chunk is a child of the previous one
                if (
                    prev_chunk_headings
                    and chunk_headings
                    and chunk_headings[0][0] > prev_chunk_headings[0][0]
                ):
                    metadata["chunks"][chunk_idx]["parent"] = prev_chunk_idx + 1

                    if "children" not in metadata["chunks"][prev_chunk_idx]:
                        metadata["chunks"][prev_chunk_idx]["children"] = []

                    metadata["chunks"][prev_chunk_idx]["children"].append(chunk_idx + 1)

            # Look for sibling relationships
            if i > 0 and chunk_headings:
                for j in range(i - 1, -1, -1):
                    prev_chunk_idx, prev_chunk_headings = all_headings[j]

                    if (
                        prev_chunk_headings
                        and chunk_headings
                        and chunk_headings[0][0] == prev_chunk_headings[0][0]
                    ):
                        if "siblings" not in metadata["chunks"][chunk_idx]:
                            metadata["chunks"][chunk_idx]["siblings"] = []

                        metadata["chunks"][chunk_idx]["siblings"].append(
                            prev_chunk_idx + 1
                        )

                        if "siblings" not in metadata["chunks"][prev_chunk_idx]:
                            metadata["chunks"][prev_chunk_idx]["siblings"] = []

                        metadata["chunks"][prev_chunk_idx]["siblings"].append(
                            chunk_idx + 1
                        )
                        break

        return metadata

    def _identify_content_types(self, chunk: str) -> List[str]:
        """
        Identify types of content present in a chunk.

        Args:
            chunk: The chunk to analyze

        Returns:
            List of content type names found in the chunk
        """
        lines = split_into_lines(chunk)
        content_types = set()

        # Check for special patterns
        in_code_block = False
        in_table = False
        has_list = False
        has_blockquote = False

        for line in lines:
            if is_code_block_delimiter(line):
                in_code_block = not in_code_block
                content_types.add("code_block")
            elif is_heading(line):
                content_types.add("heading")
            elif is_table_row(line) or is_table_separator(line):
                in_table = True
                content_types.add("table")
            elif is_list_item(line):
                has_list = True
                content_types.add("list")
            elif is_blockquote(line):
                has_blockquote = True
                content_types.add("blockquote")
            elif is_footnote_def(line):
                content_types.add("footnote")
            elif is_horizontal_rule(line):
                content_types.add("horizontal_rule")
            elif is_html_tag(line):
                content_types.add("html")
            elif is_yaml_delimiter(line):
                content_types.add("yaml")

        return list(content_types)

    def _enhance_cross_chunk_references(
        self, chunks: List[str], metadata: dict
    ) -> List[str]:
        """
        Track and enhance links that point between chunks.

        Args:
            chunks: List of markdown chunks
            metadata: Metadata about the chunks

        Returns:
            List of chunks with enhanced cross-references
        """
        # Extract all headings from all chunks with their chunk index
        heading_map = {}  # Maps heading text to chunk index

        for chunk_idx, chunk_data in enumerate(metadata["chunks"]):
            for heading in chunk_data.get("headings", []):
                heading_text = heading["text"]
                heading_map[heading_text.lower()] = chunk_idx

        # Process each chunk to find and enhance links
        enhanced_chunks = []

        # Compile the pattern once for efficiency
        link_pattern = re.compile(r"\[(.*?)\]\(#(.*?)\)")

        for chunk_idx, chunk in enumerate(chunks):
            # Define the replacement function
            def replace_link(match):
                link_text = match.group(1)
                target = match.group(2).lower()

                # Check if this target exists in another chunk
                if target in heading_map and heading_map[target] != chunk_idx:
                    target_chunk = heading_map[target] + 1  # 1-indexed in metadata
                    return f"[{link_text}](#chunk{target_chunk}:{target})"

                return match.group(0)  # Keep original if no match

            # Apply the replacement using the compiled pattern
            modified_chunk = link_pattern.sub(replace_link, chunk)
            enhanced_chunks.append(modified_chunk)

        return enhanced_chunks

    def _extract_document_title(self, markdown_text: str) -> str:
        """
        Extract the document title from the markdown content.

        Args:
            markdown_text: The markdown text to analyze

        Returns:
            The document title or a default title
        """
        # Try to extract from YAML front matter
        if is_yaml_delimiter(markdown_text.lstrip().split("\n", 1)[0]):
            try:
                # Find the YAML block
                yaml_pattern = r"^---\n(.*?)\n---"
                yaml_match = re.search(yaml_pattern, markdown_text, re.DOTALL)
                if yaml_match:
                    yaml_content = yaml_match.group(1)
                    yaml_data = yaml.safe_load(yaml_content)
                    if isinstance(yaml_data, dict) and "title" in yaml_data:
                        return yaml_data["title"]
            except Exception:
                pass  # If yaml parsing fails, continue to next method

        # Try to extract from the first heading
        lines = split_into_lines(markdown_text)
        for line in lines:
            if is_heading(line):
                return line.lstrip("#").strip()

        # Use source document name if available
        if self.source_document:
            return (
                os.path.basename(self.source_document)
                .split(".")[0]
                .replace("_", " ")
                .title()
            )

        # Default title if nothing found
        return "Untitled Document"

    def _add_metadata_to_chunks(self, chunks: List[str], metadata: dict) -> List[str]:
        """
        Embed metadata in chunks as YAML front matter.

        Args:
            chunks: List of markdown chunks
            metadata: The metadata for all chunks

        Returns:
            List of chunks with embedded metadata
        """
        chunks_with_metadata = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata["chunks"][i]
            chunk_index = i + 1  # 1-indexed for user-facing identifiers

            # Detect if the chunk already has YAML front matter
            original_yaml = None
            chunk_content = chunk

            if chunk.startswith("---"):
                try:
                    yaml_end = chunk.find("---", 3)
                    if yaml_end > 0:
                        yaml_text = chunk[3:yaml_end].strip()
                        original_yaml = yaml.safe_load(yaml_text)
                        chunk_content = chunk[yaml_end + 3 :].lstrip()
                except Exception:
                    # If parsing fails, assume it's not valid YAML front matter
                    pass

            # Build the chunk metadata
            yaml_metadata = {
                "chunk": {
                    "id": chunk_index,
                    "total": total_chunks,
                    "previous": chunk_index - 1 if chunk_index > 1 else None,
                    "next": chunk_index + 1 if chunk_index < total_chunks else None,
                    "length": len(chunk),
                    "position": f"{(chunk_index / total_chunks) * 100:.1f}%",
                },
                "document": {
                    "title": self.document_title,
                    "source": self.source_document,
                },
            }

            # Add content type information
            yaml_metadata["content"] = {
                "types": chunk_metadata.get("content_types", []),
                "word_count": len(re.findall(r"\w+", chunk)),
                "characters": len(chunk),
            }

            # Add heading information
            headings = chunk_metadata.get("headings", [])
            if headings:
                yaml_metadata["headings"] = {}

                # Find main heading
                main_heading = headings[0]["text"] if headings else None
                yaml_metadata["headings"]["main"] = main_heading

                # Find parent headings if they exist
                parent_id = chunk_metadata.get("parent")
                if parent_id:
                    parent_path = []
                    current_parent = parent_id
                    while current_parent:
                        parent_idx = current_parent - 1  # Convert to 0-indexed
                        if parent_idx < 0 or parent_idx >= len(metadata["chunks"]):
                            break

                        parent_data = metadata["chunks"][parent_idx]
                        parent_headings = parent_data.get("headings", [])
                        if parent_headings:
                            parent_path.insert(0, parent_headings[0]["text"])

                        current_parent = parent_data.get("parent")

                    yaml_metadata["headings"]["hierarchy"] = parent_path

                # Add full list of headings in the chunk
                yaml_metadata["headings"]["all"] = [h["text"] for h in headings]

            # Merge with original YAML if it exists
            if original_yaml and isinstance(original_yaml, dict):
                # Don't overwrite our metadata if there are key conflicts
                for key, value in original_yaml.items():
                    if key not in yaml_metadata:
                        yaml_metadata[key] = value

            # Convert metadata to YAML string
            yaml_string = yaml.dump(
                yaml_metadata, sort_keys=False, default_flow_style=False
            )

            # Create the new chunk with metadata
            chunk_with_metadata = f"---\n{yaml_string}---\n\n{chunk_content}"
            chunks_with_metadata.append(chunk_with_metadata)

        return chunks_with_metadata
