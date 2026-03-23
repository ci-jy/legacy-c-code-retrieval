"""Static call graph over function units."""

from __future__ import annotations

from collections import defaultdict

from .parser import FunctionUnit


class CallGraph:
    """Caller/callee edges between functions defined in the repository.

    A call `f()` inside file F is resolved by C's visibility rules, approximated
    without a preprocessor: a definition of `f` in F itself wins; otherwise every
    non-static definition of `f` (and static ones in headers, which are textually
    included) is a candidate. Calls through function pointers and calls to
    functions defined outside the repository produce no edge.
    """

    def __init__(self, units: list[FunctionUnit]):
        by_name: dict[str, list[FunctionUnit]] = defaultdict(list)
        for u in units:
            by_name[u.name].append(u)
        self._callees: dict[str, list[str]] = {u.id: [] for u in units}
        self._callers: dict[str, list[str]] = {u.id: [] for u in units}
        for u in units:
            for name in u.calls:
                for target in self._resolve(u, by_name.get(name, [])):
                    if target.id not in self._callees[u.id]:
                        self._callees[u.id].append(target.id)
                        self._callers[target.id].append(u.id)
        for d in (self._callees, self._callers):
            for k in d:
                d[k].sort()

    @staticmethod
    def _resolve(caller: FunctionUnit, candidates: list[FunctionUnit]) -> list[FunctionUnit]:
        same_file = [c for c in candidates if c.file == caller.file]
        if same_file:
            return same_file
        return [c for c in candidates if not c.is_static or c.file.endswith(".h")]

    def callees(self, unit_id: str) -> list[str]:
        return list(self._callees.get(unit_id, []))

    def callers(self, unit_id: str) -> list[str]:
        return list(self._callers.get(unit_id, []))

    def neighbours(self, unit_id: str) -> list[str]:
        return sorted(set(self._callees.get(unit_id, [])) | set(self._callers.get(unit_id, [])))

    def edges(self) -> set[tuple[str, str]]:
        return {(a, b) for a, bs in self._callees.items() for b in bs}
