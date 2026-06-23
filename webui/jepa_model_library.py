from __future__ import annotations

import re
from pathlib import Path


class JepaModelLibrary:

    OPERATIONAL_PAREN   = re.compile(r"\s*\([^()]*(?:\d{4}-\d{2}-\d{2}|referee|first[- ]pass|regression|corrected|reviewed|audit)[^()]*\)", re.IGNORECASE)
    OPERATIONAL_MARKERS = re.compile(r"\d{4}-\d{2}-\d{2}|referee|first[- ]pass|smoke[- ]test|regression|review date|reviewed|corrected on|correction|audit|flagged", re.IGNORECASE)
    OPERATIONAL_HEADING = re.compile(r"correction|checkpoint continuity|review", re.IGNORECASE)

    NOTE_FILES = {
        "jepa_profile" : "JEPA Backbone + Profile AE.md",
        "jepa_image"   : "JEPA Image AE + Backbone.md",
        "jepa_full"    : "JEPA Image AE + Backbone + Profile AE.md",
    }

    def note(self, key: str) -> dict | None:
        filename = self.NOTE_FILES.get(key)
        if filename is None:
            return None

        path = self._notes_dir() / filename
        if not path.exists():
            return None

        markdown = self._strip_operational(path.read_text(encoding="utf-8"))
        links    = {name[:-3]: model_key for model_key, name in self.NOTE_FILES.items()}
        return {"key": key, "note": filename[:-3], "markdown": markdown, "links": links}

    def _strip_operational(self, markdown: str) -> str:
        out        = []
        skip_level = None
        for line in markdown.split("\n"):
            heading = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if heading:
                level = len(heading.group(1))
                if skip_level is not None and level <= skip_level:
                    skip_level = None
                if skip_level is None and self.OPERATIONAL_HEADING.search(heading.group(2)):
                    skip_level = level
                    continue
            if skip_level is not None:
                continue

            cleaned = self.OPERATIONAL_PAREN.sub("", line)
            if self.OPERATIONAL_MARKERS.search(cleaned):
                sentences = re.split(r"(?<=[.!?])\s+", cleaned)
                kept      = [s for s in sentences if not self.OPERATIONAL_MARKERS.search(s)]
                cleaned   = " ".join(kept)
                if not cleaned.strip() or cleaned.strip() in ("|", "-", ">"):
                    continue
            out.append(cleaned)

        collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
        return collapsed

    def _notes_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parent.parent
        repo_dir  = repo_root / "notes" / "models"
        if repo_dir.is_dir():
            return repo_dir

        vault_root = repo_root.parent.parent
        return vault_root / "notes" / "DLR-TomoSAR" / "Models"

    def collect(self) -> list[dict]:
        return self._families()

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "JEPA",
                "blurb"  : "Joint-embedding predictive training: a pretrained profile autoencoder defines a target embedding space and/or a pretrained image autoencoder provides a learned front-end, while the backbone is the trainable predictor.",
                "models" : [
                    {
                        "key": "jepa_profile", "name": "Backbone + Profile AE", "skip": "Backbone + Profile AE",
                        "head": "Embedding MSE + curve recon", "params": "backbone + frozen AE", "recommended": False,
                        "when": "Predict in embedding space. The backbone head emits the profile-autoencoder embedding dimension; the frozen profile-AE encoder turns the reconstructed ground-truth profile into the target z*, and its decoder maps the prediction back to a curve for an auxiliary reconstruction loss.",
                    },
                    {
                        "key": "jepa_image", "name": "Image AE + Backbone", "skip": "Image AE + Backbone",
                        "head": "Parameter L1", "params": "frozen AE + backbone", "recommended": False,
                        "when": "Learned front-end. The frozen image-autoencoder encoder re-encodes the SAR stack before the backbone, which still regresses the Gaussian parameters directly against the supervised parameter loss; no embedding target is involved.",
                    },
                    {
                        "key": "jepa_full", "name": "Image AE + Backbone + Profile AE", "skip": "Image AE + Backbone + Profile AE",
                        "head": "Embedding MSE + curve recon", "params": "two frozen AEs + backbone", "recommended": True,
                        "when": "The full coupling. The image-autoencoder encoder feeds the backbone, the backbone predicts the profile-autoencoder embedding, and the profile autoencoder supplies both the target embedding and the decoder for curve reconstruction.",
                    },
                ],
            },
        ]
