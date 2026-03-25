"""Clone the pinned evaluation repositories into data/repos/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codesearch.repos import PINNED_REPOS, fetch  # noqa: E402

if __name__ == "__main__":
    for repo in PINNED_REPOS:
        path = fetch(repo)
        print(f"{repo.name:6s} {repo.tag:10s} {repo.commit}  -> {path}")
