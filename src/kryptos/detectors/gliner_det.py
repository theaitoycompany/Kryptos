"""GLiNER / GLiNER2 detector.

GLiNER's value here is not that it is the best general PII model - it is that
the label schema is an *input*.  That is what makes domain labels like
``teacher name`` or ``parent workplace`` possible at all; a fixed-taxonomy
tagger can only ever return PERSON for those.

Long transcripts are chunked on turn boundaries so a name is never split
across two requests.
"""

from __future__ import annotations

import logging
import re

from ..types import Detection, Document
from .base import Detector, DetectorUnavailable, register

log = logging.getLogger("kryptos.detectors.gliner")


@register("gliner")
class GLiNERDetector(Detector):
    def load(self) -> None:
        model_id = self.spec.get("model")
        if not model_id:
            raise DetectorUnavailable("no 'model' specified")
        try:
            from gliner import GLiNER
            from huggingface_hub import constants, snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise DetectorUnavailable("gliner not installed (pip install gliner)") from exc
        try:
            model_class = GLiNER
            local_only = bool(self.spec.get("local_files_only", False) or constants.HF_HUB_OFFLINE)
            if self.spec.get("backbone_model"):
                if not self.spec.get("backbone_revision"):
                    raise DetectorUnavailable("A pinned backbone revision is required")
                backbone = snapshot_download(
                    self.spec["backbone_model"],
                    revision=self.spec["backbone_revision"],
                    local_files_only=local_only,
                    allow_patterns=[
                        "config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "spm.model",
                    ],
                )

                class PinnedGLiNER(GLiNER):
                    @staticmethod
                    def _get_gliner_class(config):
                        variant = GLiNER._get_gliner_class(config)

                        class PinnedVariant(variant):
                            @classmethod
                            def _load_config(cls, config_file, **kwargs):
                                loaded = super()._load_config(config_file, **kwargs)
                                loaded.model_name = backbone
                                return loaded

                            @classmethod
                            def _load_tokenizer(cls, config, model_dir, cache_dir=None):
                                return AutoTokenizer.from_pretrained(
                                    backbone, local_files_only=True, fix_mistral_regex=False
                                )

                        return PinnedVariant

                model_class = PinnedGLiNER
            self.model = model_class.from_pretrained(
                model_id,
                revision=self.spec.get("revision"),
                local_files_only=local_only,
                strict=True,
            )
        except Exception:
            raise DetectorUnavailable("could not load configured GLiNER model") from None

        self.labels = self._resolve_labels()
        if not self.labels:
            raise DetectorUnavailable("no labels configured")
        self.threshold = float(self.spec.get("threshold", 0.3))
        self.chunk_chars = int(self.spec.get("chunk_chars", 1800))
        self.multi_label = bool(self.spec.get("multi_label", True))
        self.batch_size = int(self.spec.get("batch_size", 8))
        device = self.spec.get("device", "cpu")
        self.model.to(device)
        if self.spec.get("dtype") == "float16":
            self.model.half()
        self.model.eval()
        if self.config.get("runtime.strict_detectors", False):
            import torch

            def finite_output(_module, _args, output):
                values = [output]
                while values:
                    item = values.pop()
                    if isinstance(item, dict):
                        values.extend(item.values())
                    elif isinstance(item, (tuple, list)):
                        values.extend(item)
                    elif isinstance(item, torch.Tensor) and item.is_floating_point():
                        if not torch.isfinite(item).all().item():
                            raise RuntimeError("Non-finite GLiNER model output")

            self.model.model.register_forward_hook(finite_output)

    def _resolve_labels(self) -> list[str]:
        labels = self.spec.get("labels", [])
        if isinstance(labels, str) and labels.startswith("@"):
            labels = self.config.get("label_sets." + labels[1:], [])
        return [str(x) for x in labels or []]

    # ------------------------------------------------------------------
    def detect(self, doc: Document) -> list[Detection]:
        out: list[Detection] = []
        chunks = _chunks(doc, self.chunk_chars)
        tokenizer = self.model.data_processor.transformer_tokenizer
        token_budget = min(256, int(tokenizer.model_max_length) - 256)
        if token_budget < 64:
            raise RuntimeError("Insufficient model context for the configured labels")
        pending = list(chunks)
        chunks = []
        while pending:
            chunk, offset = pending.pop(0)
            length = len(tokenizer.encode(chunk, add_special_tokens=False, truncation=False))
            if length <= token_budget:
                chunks.append((chunk, offset))
            elif len(chunk) <= 64:
                raise RuntimeError("Input cannot be covered within model context")
            else:
                middle = len(chunk) // 2
                overlap = min(40, middle // 3)
                pending[0:0] = [
                    (chunk[: middle + overlap], offset),
                    (chunk[middle - overlap :], offset + middle - overlap),
                ]
        predictions = []
        for begin in range(0, len(chunks), self.batch_size):
            texts = [c[0] for c in chunks[begin : begin + self.batch_size]]
            preds = self.model.inference(
                texts,
                self.labels,
                batch_size=self.batch_size,
                threshold=self.threshold,
                multi_label=self.multi_label,
            )
            if len(preds) != len(texts):
                raise RuntimeError("Incomplete GLiNER batch")
            predictions.extend(preds)
        for (chunk, offset), preds in zip(chunks, predictions):
            for p in preds or []:
                entity = self.map_label(p.get("label", ""))
                if entity is None:
                    continue
                out.append(
                    self.make(
                        doc,
                        offset + int(p["start"]),
                        offset + int(p["end"]),
                        entity,
                        float(p.get("score", self.threshold)),
                        raw_label=p.get("label", ""),
                    )
                )
        return out


def _chunks(doc: Document, size: int) -> list[tuple[str, int]]:
    """Cover long turns and plain text with overlapping, bounded windows."""
    if size < 128:
        raise ValueError("GLiNER chunk size is too small")
    out = []
    start, overlap = 0, min(240, size // 4)
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        # Also bound the word count: GLiNER otherwise silently truncates very
        # dense text at its configured word limit.
        words = list(re.finditer(r"\S+", doc.text[start:end]))
        if len(words) > 180:
            end = start + words[179].end()
        if end < len(doc.text):
            boundary = doc.text.rfind(" ", start + max(1, (end - start) // 2), end)
            if boundary > start:
                end = boundary
        out.append((doc.text[start:end], start))
        if end == len(doc.text):
            break
        start = max(start + 1, end - min(overlap, (end - start) // 3))
    return out or [("", 0)]
