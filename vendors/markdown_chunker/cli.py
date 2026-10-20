#!/usr/bin/env python3
"""
Command-line interface for the Markdown Chunker.

Usage:
    markdown-chunker [options] <input_file> [<output_dir>]

Options:
    --min-chunk-len=<len>       Minimum chunk length in characters [default: 512]
    --soft-max-len=<len>        Soft maximum chunk length in characters [default: 1024]
    --hard-max-len=<len>        Hard maximum chunk length in characters [default: 2048]
    --no-headers-footers        Disable header and footer detection
    --no-duplicates             Disable duplicate detection
    --add-metadata              Embed metadata in each chunk as YAML front matter
    --document-title=<title>    Document title to use in metadata (auto-detected if not provided)
    --parallel                  Enable parallel processing for large documents
    --max-workers=<num>         Maximum number of worker processes for parallel processing
    --verbose                   Enable verbose output
    --help                      Show this help message

If output_dir is not specified, chunks will be written to the './outputs' directory.
"""

import os
import sys
import argparse
import json
from pathlib import Path

from markdown_chunker import MarkdownChunkingStrategy


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk a markdown file into smaller pieces while preserving structure."
    )
    parser.add_argument("input_file", type=str, help="The markdown file to chunk")
    parser.add_argument(
        "output_dir", type=str, nargs="?", help="Directory to output chunks"
    )
    parser.add_argument(
        "--min-chunk-len",
        type=int,
        default=512,
        help="Minimum chunk length in characters [default: 512]",
    )
    parser.add_argument(
        "--soft-max-len",
        type=int,
        default=1024,
        help="Soft maximum chunk length in characters [default: 1024]",
    )
    parser.add_argument(
        "--hard-max-len",
        type=int,
        default=2048,
        help="Hard maximum chunk length in characters [default: 2048]",
    )
    parser.add_argument(
        "--no-headers-footers",
        action="store_false",
        dest="detect_headers_footers",
        help="Disable header and footer detection",
    )
    parser.add_argument(
        "--no-duplicates",
        action="store_false",
        dest="remove_duplicates",
        help="Disable duplicate detection",
    )
    parser.add_argument(
        "--add-metadata",
        action="store_true",
        help="Embed metadata in each chunk as YAML front matter",
    )
    parser.add_argument(
        "--document-title",
        type=str,
        help="Document title to use in metadata (auto-detected if not provided)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for large documents",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes for parallel processing",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Validate arguments
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        return 1

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("./outputs")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Chunk size constraints:")
        print(f"  Minimum length: {args.min_chunk_len}")
        print(f"  Soft maximum length: {args.soft_max_len}")
        print(f"  Hard maximum length: {args.hard_max_len}")
        print(
            f"Header/footer detection: {'enabled' if args.detect_headers_footers else 'disabled'}"
        )
        print(
            f"Duplicate prevention: {'enabled' if args.remove_duplicates else 'disabled'}"
        )
        print(f"Metadata embedding: {'enabled' if args.add_metadata else 'disabled'}")
        print(f"Parallel processing: {'enabled' if args.parallel else 'disabled'}")
        if args.parallel and args.max_workers:
            print(f"Max worker processes: {args.max_workers}")
        if args.document_title:
            print(f"Document title: {args.document_title}")

    # Read the input file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 1

    # Create the chunking strategy
    strategy = MarkdownChunkingStrategy(
        min_chunk_len=args.min_chunk_len,
        soft_max_len=args.soft_max_len,
        hard_max_len=args.hard_max_len,
        detect_headers_footers=args.detect_headers_footers,
        remove_duplicates=args.remove_duplicates,
        add_metadata=args.add_metadata,
        document_title=args.document_title,
        source_document=str(input_file),
        parallel_processing=args.parallel,
        max_workers=args.max_workers,
    )

    # Chunk the markdown
    if args.verbose:
        print("Chunking markdown...")

    chunks = strategy.chunk_markdown(content)

    if args.verbose:
        print(f"Created {len(chunks)} chunks.")

    # Write chunks to files
    metadata = {
        "original_file": str(input_file),
        "total_chunks": len(chunks),
        "chunks": [],
    }

    for i, chunk in enumerate(chunks):
        chunk_file = output_dir / f"chunk_{i + 1:03d}.md"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)

        if args.verbose:
            print(
                f"Wrote chunk {i + 1}/{len(chunks)} to {chunk_file} ({len(chunk)} characters)"
            )

        metadata["chunks"].append(
            {
                "id": i + 1,
                "filename": chunk_file.name,
                "length": len(chunk),
                "first_heading": extract_first_heading(chunk),
            }
        )

    # Write metadata file
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.verbose:
        print(f"Wrote metadata to {metadata_file}")

    return 0


def extract_first_heading(text):
    """Extract the first heading from a markdown text."""
    import re

    match = re.search(r"^(#+)\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(2)
    return None


if __name__ == "__main__":
    sys.exit(main())
