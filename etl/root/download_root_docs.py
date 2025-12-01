#!/usr/bin/env python3
"""Download ROOT documentation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Download ROOT documentation using the bash script."""
    parser = argparse.ArgumentParser(description="Download ROOT documentation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="etl/root/data/root/master",
        help="Output directory for downloaded documentation (default: etl/root/data/root/master)",
    )
    args = parser.parse_args()

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bash_script = script_dir / "download_root_docs.sh"

    if not bash_script.exists():
        print(f"Error: Download script not found at {bash_script}", file=sys.stderr)
        sys.exit(1)

    # Make sure the script is executable
    os.chmod(bash_script, 0o755)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run the bash script
    print(f"Downloading ROOT documentation to {args.output_dir}...")
    try:
        result = subprocess.run([str(bash_script)], cwd=script_dir, check=True, capture_output=False)
        print("\n✅ ROOT documentation downloaded successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading ROOT documentation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
