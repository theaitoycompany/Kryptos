"""Generic Hugging Face token-classification detector.

One adapter covers every encoder-style PII tagger - OpenAI Privacy Filter,
the OpenMed multilingual fine-tune, DFKI's DialogPII model, LiquidAI's LFM2.5
encoder - because they all expose the same interface: a fast tokenizer with
offset mapping plus per-token label logits.  Only the label vocabulary
differs, and that is handled by the taxonomy's per-family label map.

The adapter is written to be safe rather than clever:
  * long documents are windowed with overlap, then spans are de-duplicated
  * both BIO and BIOES decoding are supported, including malformed sequences
    (an I- with no preceding B- still opens a span rather than being dropped)
  * the model self-disables if transformers/torch or the weights are missing
"""

from __future__ import annotations

import logging
import re

from ..types import Detection, Document
from .base import Detector, DetectorUnavailable, register

log = logging.getLogger("kryptos.detectors.hf")

_PREFIX_RE = re.compile(r"^(B|I|O|E|S|L|U)[-_]", re.I)


def split_label(label: str) -> tuple[str, str]:
    """'B-private_person' -> ('B', 'private_person')."""
    if label in ("O", "o", ""):
        return "O", ""
    m = _PREFIX_RE.match(label)
    if m:
        return m.group(1).upper(), label[m.end() :]
    return "B", label  # unprefixed label vocabularies behave like all-B


@register("hf_token")
class HFTokenDetector(Detector):
    def load(self) -> None:
        model_id = self.spec.get("model")
        if not model_id:
            raise DetectorUnavailable("no 'model' specified")
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise DetectorUnavailable(
                "transformers/torch not installed (pip install -r requirements-ml.txt)"
            ) from exc

        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        local_only = bool(self.spec.get("local_files_only", False))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                local_files_only=local_only,
                revision=self.spec.get("revision"),
                trust_remote_code=bool(self.spec.get("trust_remote_code", False)),
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                local_files_only=local_only,
                revision=self.spec.get("revision"),
                dtype=getattr(torch, self.spec["dtype"]) if self.spec.get("dtype") else None,
                trust_remote_code=bool(self.spec.get("trust_remote_code", False)),
            )
        except Exception:
            raise DetectorUnavailable("could not load configured token classifier") from None

        if not getattr(self.tokenizer, "is_fast", False):
            raise DetectorUnavailable(
                f"{model_id} has no fast tokenizer; character offsets are unavailable"
            )

        device = self.spec.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.torch = torch

        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.aggregation = self.spec.get("aggregation", "bio").lower()
        self.max_length = int(self.spec.get("max_length", 512))
        self.stride = int(self.spec.get("stride", 128))
        self.min_score = float(self.spec.get("min_score", 0.15))
        self.batch_size = int(self.spec.get("batch_size", 8))
        self._warned_labels: set = set()

    # ------------------------------------------------------------------
    def detect(self, doc: Document) -> list[Detection]:
        torch = self.torch
        text = doc.text
        if not text.strip():
            return []

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            stride=self.stride,
            padding=True,
        )
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)
        inputs = {
            k: v.to(self.device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }

        detections: list[Detection] = []
        n_windows = inputs["input_ids"].shape[0]
        with torch.no_grad():
            for i in range(0, n_windows, self.batch_size):
                batch = {k: v[i : i + self.batch_size] for k, v in inputs.items()}
                logits = self.model(**batch).logits
                if (
                    self.config.get("runtime.strict_detectors", False)
                    and not torch.isfinite(logits).all().item()
                ):
                    raise RuntimeError("Non-finite model logits")
                probs = torch.softmax(logits, dim=-1).cpu()
                for w in range(probs.shape[0]):
                    detections.extend(
                        self._decode_window(
                            doc, probs[w], offsets[i + w], batch["attention_mask"][w].cpu()
                        )
                    )
        return _dedupe(detections)

    # ------------------------------------------------------------------
    def _decode_window(self, doc: Document, probs, offsets, mask) -> list[Detection]:
        best_scores, best_ids = probs.max(dim=-1)
        out: list[Detection] = []
        cur_label = ""
        cur_start = -1
        cur_end = -1
        cur_scores: list[float] = []

        def close() -> None:
            nonlocal cur_label, cur_start, cur_end, cur_scores
            if cur_label and cur_start >= 0 and cur_end > cur_start and cur_scores:
                score = sum(cur_scores) / len(cur_scores)
                entity = self.map_label(cur_label)
                if entity is None:
                    if cur_label not in self._warned_labels:
                        self._warned_labels.add(cur_label)
                        log.debug("%s: unmapped label %r", self.name, cur_label)
                    if not self.config.get("fusion.drop_unmapped", True):
                        entity = "PERSON_NAME"
                if entity and score >= self.min_score:
                    out.append(
                        self.make(
                            doc, cur_start, cur_end, entity, float(score), raw_label=cur_label
                        )
                    )
            cur_label, cur_start, cur_end, cur_scores = "", -1, -1, []

        for idx in range(len(best_ids)):
            if int(mask[idx]) == 0:
                continue
            start, end = int(offsets[idx][0]), int(offsets[idx][1])
            if end <= start:  # special token
                continue
            label = self.id2label.get(int(best_ids[idx]), "O")
            prefix, body = split_label(label)
            score = float(best_scores[idx])

            if prefix == "O" or not body:
                close()
                continue
            if prefix in ("B", "S", "U") or body != cur_label:
                close()
                cur_label, cur_start, cur_end, cur_scores = body, start, end, [score]
            else:  # I / E / L continuation
                cur_end = end
                cur_scores.append(score)
            if prefix in ("S", "U", "E", "L"):
                close()
        close()
        return out


def _dedupe(dets: list[Detection]) -> list[Detection]:
    """Overlapping windows produce duplicates; keep the best-scoring copy."""
    best: dict[tuple[int, int, str], Detection] = {}
    for d in dets:
        k = d.key()
        if k not in best or d.score > best[k].score:
            best[k] = d
    return list(best.values())
