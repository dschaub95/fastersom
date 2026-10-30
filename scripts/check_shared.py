"""Check that the files kept byte-identical in fastersom and its sister k-means package match another checkout.

    python scripts/check_shared.py PATH_TO_THE_OTHER_CHECKOUT

Compares src/<package>/_shared.py, tests/test_shared.py and this script with those of the other checkout, prints a
unified diff (the other checkout's file to this one's) of each that differs, and exits with 1 if any does.
"""

import difflib
import sys
from pathlib import Path

FILES = ("src/*/_shared.py", "tests/test_shared.py", "scripts/check_shared.py")


def main(other: str) -> int:
    """Compare the files of this checkout with those of the other one; 0 if all are identical, else 1."""
    roots = Path(other).resolve(), Path(__file__).resolve().parents[1]
    differ = False
    for pattern in FILES:
        theirs, mine = (next(root.glob(pattern), root / pattern) for root in roots)  # a missing file raises below
        if theirs.read_bytes() == mine.read_bytes():
            continue
        differ = True
        lines = (path.read_text().splitlines(keepends=True) for path in (theirs, mine))
        sys.stdout.writelines(difflib.unified_diff(*lines, fromfile=str(theirs), tofile=str(mine)))
    print("differ" if differ else "identical", file=sys.stderr)
    return int(differ)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
