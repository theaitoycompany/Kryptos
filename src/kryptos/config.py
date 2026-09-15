"""Configuration: entity taxonomy, policy, detector wiring.

Config is JSON (YAML is accepted when PyYAML happens to be installed) so the
core has no dependencies.  ``Config.load()`` merges a user file over the
shipped defaults, key by key, so a caller only states what they change.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PKG_ROOT, "resources")
TAXONOMY_PATH = os.path.join(CONFIG_DIR, "taxonomy.json")
DEFAULT_POLICY_PATH = os.path.join(CONFIG_DIR, "default.json")


def _read(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional
                raise RuntimeError(f"PyYAML is required to read {path}; use JSON instead") from exc
            return yaml.safe_load(fh) or {}
        return json.load(fh)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class EntityPolicy:
    name: str
    tier: str
    action: str
    prefix: str
    priority: int
    generalized: str = ""
    threshold: float = 0.0

    @property
    def is_direct(self) -> bool:
        return self.tier == "direct"


@dataclass
class Taxonomy:
    entities: dict[str, EntityPolicy]
    label_maps: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: str = TAXONOMY_PATH) -> Taxonomy:
        raw = _read(path)
        entities = {}
        for name, spec in raw["entities"].items():
            entities[name] = EntityPolicy(
                name=name,
                tier=spec.get("tier", "quasi"),
                action=spec.get("action", "pseudonymize"),
                prefix=spec.get("prefix", name),
                priority=int(spec.get("priority", 50)),
                generalized=spec.get("generalized", ""),
            )
        maps = {
            k: {kk.lower(): vv for kk, vv in v.items()}
            for k, v in raw.get("label_maps", {}).items()
        }
        return cls(entities=entities, label_maps=maps)

    def policy(self, entity: str) -> EntityPolicy:
        if entity in self.entities:
            return self.entities[entity]
        # unknown canonical label -> treat conservatively as a direct identifier
        return EntityPolicy(
            name=entity,
            tier="direct",
            action="redact",
            prefix=entity,
            priority=50,
            generalized=f"<{entity}>",
        )

    def map_label(self, detector_family: str, raw_label: str) -> str | None:
        table = self.label_maps.get(detector_family, {})
        return table.get((raw_label or "").strip().lower())

    def priority(self, entity: str) -> int:
        return self.policy(entity).priority


@dataclass
class Config:
    raw: dict[str, Any]
    taxonomy: Taxonomy

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
        base = _read(DEFAULT_POLICY_PATH)
        if path:
            base = deep_merge(base, _read(path))
        if overrides:
            base = deep_merge(base, overrides)
        tax_path = base.get("taxonomy_path") or TAXONOMY_PATH
        if not os.path.isabs(tax_path):
            parent = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
            tax_path = os.path.join(parent, tax_path)
        return cls(raw=base, taxonomy=Taxonomy.load(tax_path))

    # ------------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    # convenience accessors used all over the pipeline -----------------
    @property
    def detectors(self) -> list[dict[str, Any]]:
        return [d for d in self.raw.get("detectors", []) if d.get("enabled", True)]

    @property
    def known_values(self) -> dict[str, Any]:
        return self.raw.get("known_values", {})

    def entity_threshold(self, entity: str) -> float:
        table = self.raw.get("thresholds", {})
        if entity in table:
            return float(table[entity])
        tier = self.taxonomy.policy(entity).tier
        return float(table.get("_tier_" + tier, table.get("_default", 0.35)))

    def action_for(self, entity: str) -> str:
        override = self.raw.get("actions", {}).get(entity)
        return override or self.taxonomy.policy(entity).action
