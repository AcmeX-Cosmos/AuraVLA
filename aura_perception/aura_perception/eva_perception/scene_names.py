from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any


DEFAULT_SCENE_OBJECTS: dict[str, dict[str, Any]] = {
    "banana": {
        "prim_path": "/World/banana",
        "aliases": ["香蕉", "黄色香蕉", "yellow banana"],
    },
    "basket": {
        "prim_path": "/World/small_KLT_visual",
        "aliases": [
            "篮子",
            "盒子",
            "箱子",
            "收纳箱",
            "紫色盒子",
            "紫色箱子",
            "紫色收纳箱",
            "purple box",
            "purple bin",
            "container",
        ],
    },
    "scissors": {
        "prim_path": "/World/_37_scissors",
        "aliases": ["剪刀", "scissor"],
    },
    "master_chef_can": {
        "prim_path": "/World/_02_master_chef_can",
        "aliases": [
            "MasterChef",
            "MasterChef can",
            "MasterChef罐头",
            "MasterChef 罐头",
            "blue can",
            "blue coffee can",
            "coffee can",
            "咖啡罐",
            "蓝色罐头",
            "蓝色咖啡罐",
        ],
    },
    "tomato_soup_can": {
        "prim_path": "/World/_05_tomato_soup_can",
        "aliases": [
            "tomato soup can",
            "red can",
            "番茄汤罐",
            "番茄汤罐头",
            "红色罐头",
            "红罐头",
            "红白罐头",
            "红白相间罐头",
        ],
    },
    "mug": {
        "prim_path": "/World/SM_Mug_A2",
        "aliases": ["杯子", "马克杯", "绿色杯子", "绿色马克杯", "green mug"],
    },
}


def _normalized_name(value: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "").strip().lower())


@dataclass(frozen=True)
class SceneObjectName:
    canonical_name: str
    prim_path: str
    aliases: tuple[str, ...]


class SceneNameResolver:
    def __init__(self, objects: Sequence[SceneObjectName]) -> None:
        self._objects = tuple(objects)
        self._aliases: dict[str, str] = {}
        self._by_canonical: dict[str, SceneObjectName] = {}
        for item in self._objects:
            self._by_canonical[item.canonical_name] = item
            for alias in (item.canonical_name, item.prim_path, *item.aliases):
                normalized = _normalized_name(alias)
                if normalized:
                    self._aliases[normalized] = item.canonical_name
    def build_alias_hints(self) -> str:
        """Return a compact alias-mapping hint string for the NVIDIA VLM prompt."""
        lines = []
        for canonical_name in sorted(self._by_canonical):
            item = self._by_canonical[canonical_name]
            hint_keywords = [item.canonical_name] + list(item.aliases)
            deduped = list(dict.fromkeys(hint_keywords))
            lines.append("  " + " / ".join(deduped))
        if not lines:
            return ""
        return (
            "The following object names are recognized by the robot. "
            "When the user mentions any listed alias, match it to the closest visible "
            "object by color/shape even if the exact word differs:\n"
            + "\n".join(lines)
        )

    @classmethod
    def from_mapping(
        cls,
        settings: Mapping[str, Any] | None = None,
    ) -> "SceneNameResolver":
        configured = (settings or {}).get("objects", settings or {})
        merged = {name: dict(value) for name, value in DEFAULT_SCENE_OBJECTS.items()}
        if isinstance(configured, Mapping):
            for name, value in configured.items():
                if not isinstance(value, Mapping):
                    continue
                current = merged.setdefault(str(name), {})
                configured_aliases = value.get("aliases")
                if configured_aliases is not None:
                    if isinstance(configured_aliases, str):
                        configured_aliases = [configured_aliases]
                    current["aliases"] = list(
                        dict.fromkeys(
                            [
                                *(current.get("aliases") or []),
                                *configured_aliases,
                            ]
                        )
                    )
                current.update(
                    {key: nested for key, nested in value.items() if key != "aliases"}
                )

        objects = []
        for canonical_name, value in merged.items():
            prim_path = str(value.get("prim_path") or "").strip()
            if not prim_path:
                continue
            aliases = value.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            objects.append(
                SceneObjectName(
                    canonical_name=str(canonical_name).strip(),
                    prim_path=prim_path,
                    aliases=tuple(str(alias).strip() for alias in aliases if str(alias).strip()),
                )
            )
        return cls(objects)

    def canonicalize(self, value: Any) -> str:
        original = str(value or "").strip()
        normalized = _normalized_name(original)
        if not normalized:
            return original
        exact = self._aliases.get(normalized)
        if exact is not None:
            return exact
        matches = [
            (len(alias), canonical)
            for alias, canonical in self._aliases.items()
            if len(alias) >= 2 and alias in normalized
        ]
        return max(matches)[1] if matches else original

    def resolve_traversal_names(self, canonical_name: str) -> frozenset[str]:
        """Return lowercased names for stage-traversal fallback lookup.

        Includes the canonical name, all registered aliases, and the last
        path segment of the registered prim_path (e.g. ``_02_master_chef_can``
        for ``/World/_02_master_chef_can``), so the traversal succeeds even
        when the prim's USD name contains a numeric prefix.
        """
        item = self._by_canonical.get(canonical_name)
        if item is None:
            return frozenset((canonical_name.lower(),))
        names: set[str] = {canonical_name, *item.aliases}
        path_suffix = item.prim_path.rstrip("/").rsplit("/", 1)[-1]
        if path_suffix:
            names.add(path_suffix)
        return frozenset(n.lower() for n in names)

    def prim_candidates(self, value: Any) -> tuple[str, ...]:
        original = str(value or "").strip()
        candidates = []
        if original.startswith("/"):
            candidates.append(original)
        canonical = self.canonicalize(original)
        item = self._by_canonical.get(canonical)
        if item is not None:
            candidates.append(item.prim_path)
        if original and not original.startswith("/"):
            candidates.extend((f"/World/{original}", f"/World/{original.lower()}"))
        return tuple(dict.fromkeys(candidates))


DEFAULT_SCENE_NAME_RESOLVER = SceneNameResolver.from_mapping()
