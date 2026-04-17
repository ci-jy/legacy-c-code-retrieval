"""Open-source C repositories used for evaluation, pinned to exact commits."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "repos"


@dataclass(frozen=True)
class PinnedRepo:
    name: str
    url: str
    commit: str
    tag: str
    license: str
    # Sub-directories to read (empty: the whole repository). Used to leave out vendored code.
    paths: tuple[str, ...] = ()

    def source_dirs(self, root: Path) -> list[Path]:
        return [root / p for p in self.paths] if self.paths else [root]


PINNED_REPOS = (
    PinnedRepo("lua", "https://github.com/lua/lua.git", "312b9efaa1061c2c4cad08554dbc1351c3270eef", "v5.4.9", "MIT"),
    PinnedRepo("zlib", "https://github.com/madler/zlib.git", "da607da739fa6047df13e66a2af6b8bec7c2a498", "v1.3.2",
               "zlib"),
    PinnedRepo("jq", "https://github.com/jqlang/jq.git", "34f7186b86743a083a589741b6cea95293524108", "jq-1.8.2",
               "MIT"),
)


# Training corpora for fine-tuning the embedding model. They must stay disjoint from
# PINNED_REPOS: Redis vendors Lua under deps/, so only its own src/ is read.
TRAINING_REPOS = (
    PinnedRepo("sqlite", "https://github.com/sqlite/sqlite.git", "262de1bebb0647eb6fa6a2b0434111c7d831a14d",
               "version-3.47.2", "public domain", ("src",)),
    PinnedRepo("redis", "https://github.com/redis/redis.git", "a0a6f23d997b024689ba157916837f493a593a34", "7.4.2",
               "BSD-3-Clause", ("src",)),
    PinnedRepo("curl", "https://github.com/curl/curl.git", "75a2079d5c28debb2eaa848ca9430f1fe0d7844c",
               "curl-8_11_1", "curl", ("lib", "src")),
)


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch(repo: PinnedRepo, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Shallow-fetch repo at its pinned commit into data_dir/<name> (reused when already there)."""
    dest = data_dir / repo.name
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", cwd=dest)
    if _head(dest) != repo.commit:
        _git("fetch", "-q", "--depth", "1", repo.url, repo.commit, cwd=dest)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=dest)
    head = _head(dest)
    if head != repo.commit:
        raise RuntimeError(f"{repo.name}: expected {repo.commit}, got {head}")
    return dest


def _head(dest: Path) -> str:
    try:
        return _git("rev-parse", "--verify", "-q", "HEAD", cwd=dest)
    except subprocess.CalledProcessError:
        return ""


def fetch_all(data_dir: Path = DEFAULT_DATA_DIR, repos=PINNED_REPOS) -> dict[str, Path]:
    return {r.name: fetch(r, data_dir) for r in repos}
