"""Convert .py notebooks (percent format) to .ipynb for Colab."""
import json
import os
import re
import glob


def py_to_ipynb(py_path: str) -> str:
    """Convert a percent-format .py file to .ipynb JSON."""
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into cells by '# %%' markers
    raw_cells = re.split(r'^# %%', content, flags=re.MULTILINE)

    cells = []
    for i, cell_content in enumerate(raw_cells):
        if i == 0 and not cell_content.strip():
            continue  # Skip empty preamble

        cell_content = cell_content.rstrip() + "\n"

        # Check if it's a markdown cell
        if cell_content.startswith(" [markdown]"):
            cell_content = cell_content.replace(" [markdown]", "", 1).strip()
            # Remove leading '# ' from each line (markdown comment syntax)
            lines = []
            for line in cell_content.split("\n"):
                if line.startswith("# "):
                    lines.append(line[2:] + "\n")
                elif line == "#":
                    lines.append("\n")
                else:
                    lines.append(line + "\n")
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines,
            })
        else:
            # Code cell — strip leading newline
            code = cell_content.lstrip("\n")
            lines = [line + "\n" for line in code.rstrip("\n").split("\n")]
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": lines,
                "outputs": [],
                "execution_count": None,
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return json.dumps(notebook, indent=1, ensure_ascii=False)


def main():
    py_files = sorted(glob.glob("notebooks/*.py"))
    print(f"Found {len(py_files)} .py notebooks to convert\n")

    for py_path in py_files:
        ipynb_path = py_path.replace(".py", ".ipynb")
        ipynb_content = py_to_ipynb(py_path)

        with open(ipynb_path, "w", encoding="utf-8") as f:
            f.write(ipynb_content)

        # Count cells
        nb = json.loads(ipynb_content)
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"  [OK] {os.path.basename(ipynb_path)} ({n_code} code + {n_md} markdown cells)")

    print(f"\nDone! {len(py_files)} notebooks converted to .ipynb")


if __name__ == "__main__":
    main()
