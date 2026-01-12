#!/usr/bin/env python3
"""
Bundle code and text files into a JSON file for easier sharing.

This script:
- Respects .gitignore rules using gitignore-parser
- Only includes code and text files (no binaries/images)
- Creates a JSON bundle with file paths as keys and file contents as values
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Set

try:
    from gitignore_parser import parse_gitignore
except ImportError:
    print("Error: gitignore-parser is not installed.")
    print("Install it with: pip install gitignore-parser")
    sys.exit(1)


# Text/code file extensions to include
TEXT_EXTENSIONS: Set[str] = {
    # Code files
    '.py', '.ts', '.js', '.tsx', '.jsx',
    # Config files
    '.json', '.toml', '.yaml', '.yml', '.ini', '.cfg', '.conf',
    # Markup/Text
    '.html', '.htm', '.css', '.md', '.txt', '.rst',
    # Shell scripts
    '.sh', '.bash', '.zsh', '.fish',
    # Other text formats
    '.xml', '.svg',  # SVG is text-based
    '.csv', '.tsv',
}


def is_text_file(filepath: Path) -> bool:
    """Check if a file is a text/code file based on extension."""
    # Check if file has a known text extension
    ext = filepath.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return True
    
    # Check for files without extension (like LICENSE, README, etc.)
    name_upper = filepath.name.upper()
    if name_upper in TEXT_EXTENSIONS:
        return True
    
    # Common text files without extensions
    if name_upper in {'LICENSE', 'LICENCE', 'README', 'CHANGELOG', 'CONTRIBUTING', 'AUTHORS'}:
        return True
    
    return False


def is_binary_file(filepath: Path) -> bool:
    """Try to detect if a file is binary by reading first bytes."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(8192)  # Read first 8KB
            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True
            # Check if file is mostly printable ASCII/UTF-8
            try:
                chunk.decode('utf-8')
            except UnicodeDecodeError:
                return True
    except Exception:
        return True  # Assume binary if we can't read it
    
    return False


def collect_files(root_dir: Path, gitignore_matcher) -> Dict[str, str]:
    """
    Collect all text files that are not ignored by .gitignore.
    
    Returns a dictionary mapping relative file paths to file contents.
    """
    bundled_files: Dict[str, str] = {}
    
    for root, dirs, files in os.walk(root_dir):
        # Filter out ignored directories before walking into them
        # Use relative paths for gitignore matching
        dirs[:] = [
            d for d in dirs 
            if not gitignore_matcher(str(Path(root).relative_to(root_dir) / d))
        ]
        
        for file in files:
            filepath = Path(root) / file
            rel_path = filepath.relative_to(root_dir)
            rel_path_str = str(rel_path).replace('\\', '/')  # Normalize to forward slashes
            
            # Check if file is ignored (use relative path from project root)
            # gitignore-parser expects paths relative to .gitignore location
            if gitignore_matcher(rel_path_str):
                continue
            
            # Check if it's a text file
            if not is_text_file(filepath):
                continue
            
            # Double-check it's not binary
            if is_binary_file(filepath):
                continue
            
            # Try to read the file
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    bundled_files[rel_path_str] = content
            except Exception as e:
                print(f"Warning: Skipping {rel_path_str}: {e}", file=sys.stderr)
                continue
    
    return bundled_files


def main():
    """Main function to bundle code into JSON."""
    # Get the project root (directory containing .gitignore)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir
    
    # Find .gitignore file
    gitignore_path = project_root / '.gitignore'
    if not gitignore_path.exists():
        print(f"Warning: .gitignore not found at {gitignore_path}")
        print("Proceeding without .gitignore rules...")
        gitignore_matcher = lambda path: False
    else:
        print(f"Using .gitignore from: {gitignore_path}")
        gitignore_matcher = parse_gitignore(gitignore_path)
    
    # Collect files
    print(f"Scanning project directory: {project_root}")
    print("Collecting code and text files...")
    bundled_files = collect_files(project_root, gitignore_matcher)
    
    # Create output structure
    output = {
        'project_root': str(project_root),
        'total_files': len(bundled_files),
        'files': bundled_files
    }
    
    # Write to JSON file
    output_file = project_root / 'bundled_code.json'
    print(f"\nWriting {len(bundled_files)} files to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Calculate size
    file_size = output_file.stat().st_size
    size_mb = file_size / (1024 * 1024)
    
    print(f"✓ Successfully bundled {len(bundled_files)} files")
    print(f"✓ Output file: {output_file}")
    print(f"✓ File size: {size_mb:.2f} MB")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

