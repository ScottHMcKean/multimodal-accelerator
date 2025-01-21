#!/usr/bin/env python3
"""Export Poetry dependencies to requirements files."""
import subprocess
from pathlib import Path

def export_requirements():
    """Export Poetry dependencies to requirements.txt and requirements-dev.txt."""
    project_root = Path(__file__).parent.parent
    
    # Export main requirements (only explicit dependencies)
    subprocess.run([
        "poetry", "export",
        "-f", "requirements.txt",
        "--output", str(project_root / "requirements.txt"),
        "--without-hashes",
        "--without-unbound"  # Only include explicitly defined dependencies
    ], check=True)
    
    # Export dev requirements (only explicit dependencies)
    subprocess.run([
        "poetry", "export",
        "-f", "requirements.txt",
        "--output", str(project_root / "requirements-dev.txt"),
        "--with", "dev",
        "--without-hashes",
        "--without-unbound"  # Only include explicitly defined dependencies
    ], check=True)

if __name__ == "__main__":
    export_requirements() 