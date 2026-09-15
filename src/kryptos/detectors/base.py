"""Detector interface + registry.

Every detector receives the *original, untouched* document and returns
``Detection`` objects with character offsets.  Detectors never mutate the
text and never see each other's output - sequential redaction would destroy
the context later detectors need ("my teacher is [PII]" is unclassifiable).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from ..config import Config
from ..types import Detection, Document

log = logging.getLogger("kryptos.detectors")

_REGISTRY: dict[str, type[Detector]] = {}


def register(type_name: str) -> Callable[[type[Detector]], type[Detector]]:
    def deco(cls: type[Detector]) -> type[Detector]:
        _REGISTRY[type_name] = cls
        cls.type_name = type_name
        return cls

    return deco


class DetectorUnavailable(RuntimeError):
    """Raised when a detector's optional dependency or weights are missing."""


class Detector:
    type_name = "base"

    def __init__(self, spec: dict[str, Any], config: Config):
        self.spec = spec
        self.config = config
        self.name = spec.get("name", self.type_name)
        self.family = spec.get("family", self.type_name)
        self.weight = float(spec.get("weight", 1.0))
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Detector weight must be finite and nonnegative")
        self.available = True
        self.unavailable_reason = ""

    # -- lifecycle ------------------------------------------------------
    def load(self) -> None:
        """Acquire heavy resources.  Raise DetectorUnavailable to self-disable."""

    def detect(self, doc: Document) -> list[Detection]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers --------------------------------------------------------
    def map_label(self, raw_label: str) -> str | None:
        """Map a detector-native label to the canonical taxonomy."""
        canonical = self.config.taxonomy.map_label(self.family, raw_label)
        if canonical is None:
            # a label that is already canonical passes through unchanged
            upper = (raw_label or "").strip().upper().replace(" ", "_")
            if upper in self.config.taxonomy.entities:
                return upper
        if canonical == "KEEP":
            return None
        return canonical

    def make(
        self,
        doc: Document,
        start: int,
        end: int,
        entity: str,
        score: float,
        raw_label: str = "",
        **meta: Any,
    ) -> Detection:
        if self.config.get("runtime.strict_detectors", True) and not math.isfinite(score):
            raise RuntimeError("Non-finite detector score")
        return Detection(
            start=start,
            end=end,
            entity=entity,
            score=max(0.0, min(1.0, score * self.weight)),
            detector=self.name,
            raw_label=raw_label or entity,
            text=doc.text[start:end],
            meta=meta,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_name,
            "family": self.family,
            "available": self.available,
            "reason": self.unavailable_reason,
            "model": self.spec.get("model", ""),
            "revision": self.spec.get("revision", ""),
            "backbone_model": self.spec.get("backbone_model", ""),
            "backbone_revision": self.spec.get("backbone_revision", ""),
        }


def build_detectors(config: Config, only: list[str] | None = None) -> list[Detector]:
    """Instantiate enabled detectors; strict mode requires every one to load."""
    from . import (  # noqa: F401
        context_rules,
        gliner_det,
        hf_token,
        known_values,
        regex_rules,
    )

    names = [spec.get("name", spec.get("type")) for spec in config.detectors]
    if len(names) != len(set(names)):
        raise ValueError("Detector names must be unique")
    if only and set(only) - set(names):
        raise ValueError("Requested detector is not enabled")
    built: list[Detector] = []
    for spec in config.detectors:
        if only and spec.get("name") not in only:
            continue
        cls = _REGISTRY.get(spec.get("type", ""))
        if cls is None:
            if config.get("runtime.strict_detectors", True):
                raise DetectorUnavailable("Required detector type is unavailable")
            log.warning(
                "unknown detector type %r (detector %s) - skipped",
                spec.get("type"),
                spec.get("name"),
            )
            continue
        det = cls(spec, config)
        try:
            det.load()
        except DetectorUnavailable as exc:
            if config.get("runtime.strict_detectors", True):
                raise DetectorUnavailable("Required detector failed to load: " + det.name) from None
            det.available = False
            det.unavailable_reason = type(exc).__name__
            log.warning("detector %s disabled", det.name)
        except Exception as exc:  # defensive: a broken model must not kill the run
            if config.get("runtime.strict_detectors", True):
                raise DetectorUnavailable("Required detector failed to load: " + det.name) from None
            det.available = False
            det.unavailable_reason = type(exc).__name__
            log.warning("detector %s failed to load", det.name)
        built.append(det)
    return built


def run_detectors(
    detectors: list[Detector], doc: Document, parallel: bool = False
) -> list[Detection]:
    """Run all detectors over the same original text and pool their spans."""
    live = [d for d in detectors if d.available]
    results: list[Detection] = []
    if parallel and len(live) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(live))) as pool:
            futures = {pool.submit(_safe_detect, d, doc): d for d in live}
            for fut in futures:
                results.extend(fut.result())
    else:
        for d in live:
            results.extend(_safe_detect(d, doc))
    return results


def _safe_detect(detector: Detector, doc: Document) -> list[Detection]:
    try:
        out = detector.detect(doc) or []
    except Exception:
        if detector.config.get("runtime.strict_detectors", True):
            raise RuntimeError(
                "Required detector failed during detection: " + detector.name
            ) from None
        log.warning("detector %s raised during detection", detector.name)
        return []
    clean = []
    for d in out:
        if detector.config.get("runtime.strict_detectors", True) and not math.isfinite(d.score):
            raise RuntimeError("Non-finite detector score")
        if detector.config.get("runtime.strict_detectors", True) and not (
            0 <= d.start < d.end <= len(doc.text)
        ):
            raise RuntimeError("Invalid detector span offsets")
        d.start = max(0, min(len(doc.text), d.start))
        d.end = max(0, min(len(doc.text), d.end))
        if d.end > d.start:
            if not d.text:
                d.text = doc.text[d.start : d.end]
            clean.append(d)
    return clean
