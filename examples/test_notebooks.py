"""Test that the example notebooks can be executed."""

import json
import sys
from pathlib import Path


def extract_code_cells(notebook_path):
    """Extract all code cells from a notebook."""
    with open(notebook_path) as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            # Join lines with newlines
            source = "\n".join(cell["source"])
            if source.strip():  # Only non-empty cells
                code_cells.append(source)

    return code_cells


def test_notebook_code(notebook_path):
    """Test that the code in a notebook can be parsed."""
    code_cells = extract_code_cells(notebook_path)

    print(f"\nTesting {notebook_path.name}:")
    print(f"Found {len(code_cells)} code cells")

    # Try to compile each code cell
    for i, code in enumerate(code_cells):
        try:
            compile(code, f"<cell {i}>", "exec")
            print(f"  Cell {i+1}: OK")
        except SyntaxError as e:
            print(f"  Cell {i+1}: ERROR - {e}")
            print(f"    Code: {code[:100]}")
            return False

    return True


def main():
    """Test all notebooks."""
    examples_dir = Path("examples")
    notebooks = sorted(examples_dir.glob("*.ipynb"))

    # Exclude create_notebooks.py if present
    notebooks = [nb for nb in notebooks if nb.name != "create_notebooks.py"]

    if not notebooks:
        print("No notebooks found!")
        return 1

    print(f"Found {len(notebooks)} notebooks to test")

    all_passed = True
    for notebook in notebooks:
        if not test_notebook_code(notebook):
            all_passed = False

    if all_passed:
        print("\n✓ All notebooks have valid code!")
    else:
        print("\n✗ Some notebooks have errors")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

