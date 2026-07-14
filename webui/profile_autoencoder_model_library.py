from __future__ import annotations

import re
import sys
from pathlib import Path


class ProfileAutoencoderModelLibrary:

    OPERATIONAL_PAREN   = re.compile(r"\s*\([^()]*(?:\d{4}-\d{2}-\d{2}|referee|first[- ]pass|regression|corrected|reviewed|audit)[^()]*\)", re.IGNORECASE)
    OPERATIONAL_MARKERS = re.compile(r"\d{4}-\d{2}-\d{2}|referee|first[- ]pass|smoke[- ]test|regression|review date|reviewed|corrected on|correction|audit|flagged", re.IGNORECASE)
    OPERATIONAL_HEADING = re.compile(r"correction|checkpoint continuity|review", re.IGNORECASE)

    CONFIG_CLASSES = {
        "mlp_ae"           : "MlpAutoencoderConfig",
        "conv1d_ae"        : "Conv1dAutoencoderConfig",
        "transformer1d_ae" : "Transformer1dAutoencoderConfig",
        "resmlp_ae"        : "ResMlpAutoencoderConfig",
        "tcn_ae"           : "TcnAutoencoderConfig",
        "gru_ae"           : "GruAutoencoderConfig",
        "cnn_attn_ae"      : "CnnAttnAutoencoderConfig",
    }

    NOTE_FILES = {
        "mlp_ae"           : "MLP Autoencoder.md",
        "conv1d_ae"        : "Conv1D Autoencoder.md",
        "transformer1d_ae" : "Transformer1D Autoencoder.md",
        "resmlp_ae"        : "ResMLP Autoencoder.md",
        "tcn_ae"           : "TCN Autoencoder.md",
        "gru_ae"           : "GRU Autoencoder.md",
        "cnn_attn_ae"      : "Conv-Attention Autoencoder.md",
    }

    FALLBACK_ACTIVATION    = "gelu"
    FALLBACK_NORMALIZATION = "none"

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
        families = self._families()
        defaults = self._resolve_defaults()

        for family in families:
            for model in family["models"]:
                activation, normalization = defaults[model["key"]]
                model["activation"]    = activation
                model["normalization"] = normalization

        return families

    def _resolve_defaults(self) -> dict[str, tuple[str, str]]:
        module   = self._import_profile_autoencoder_models_config()
        resolved = {}

        for key, class_name in self.CONFIG_CLASSES.items():
            config        = getattr(module, class_name)()
            activation    = getattr(config, "activation", self.FALLBACK_ACTIVATION)
            normalization = getattr(config, "embedding_norm", self.FALLBACK_NORMALIZATION)
            resolved[key] = (str(activation).lower(), str(normalization).lower())

        return resolved

    def _import_profile_autoencoder_models_config(self):
        config_dir = Path(__file__).resolve().parent.parent / "configuration" / "architectures"
        path       = str(config_dir)

        if path not in sys.path:
            sys.path.insert(0, path)

        import profile_autoencoder

        return profile_autoencoder

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "Profile autoencoder",
                "blurb"  : "Compress the fitted, normalized elevation profile into a latent embedding and reconstruct it. The encoder defines the output latent space later predicted by JEPA.",
                "models" : [
                    {
                        "key": "mlp_ae", "name": "MLP Autoencoder", "skip": "Symmetric MLP",
                        "head": "Dense to embedding", "params": "~1.86M", "recommended": True,
                        "when": "The default starting point. Treats the profile as a flat vector; a symmetric dense encoder and decoder compress it to the embedding and reconstruct it. Cheapest and strongest baseline.",
                    },
                    {
                        "key": "conv1d_ae", "name": "Conv1D Autoencoder", "skip": "1D convolutions",
                        "head": "Conv stack to embedding", "params": "~1.60M", "recommended": False,
                        "when": "Exploits the local smoothness of the elevation profile. Stacked 1D convolutions over the range axis capture neighbouring-bin correlations before pooling to the embedding.",
                    },
                    {
                        "key": "transformer1d_ae", "name": "Transformer1D Autoencoder", "skip": "Self-attention",
                        "head": "Transformer to embedding", "params": "~1.89M", "recommended": False,
                        "when": "Long-range dependencies along the profile. A self-attention encoder and decoder model interactions between distant elevation bins, at a higher parameter cost.",
                    },
                    {
                        "key": "resmlp_ae", "name": "ResMLP Autoencoder", "skip": "Residual MLP",
                        "head": "Dense to embedding", "params": "~1.99M", "recommended": False,
                        "when": "A deeper dense alternative to the MLP baseline. Pre-norm residual blocks let the encoder and decoder go deeper without optimisation trouble, trading parameters for capacity while keeping the flat-vector treatment of the profile.",
                    },
                    {
                        "key": "tcn_ae", "name": "TCN Autoencoder", "skip": "Dilated 1D conv",
                        "head": "Dilated conv to embedding", "params": "~1.61M", "recommended": False,
                        "when": "Multi-scale local structure. Stacked dilated residual convolutions grow the receptive field exponentially over the range axis, capturing both narrow and broad elevation features without the cost of attention.",
                    },
                    {
                        "key": "gru_ae", "name": "GRU Autoencoder", "skip": "Recurrent",
                        "head": "BiGRU to embedding", "params": "~1.83M", "recommended": False,
                        "when": "A compact recurrent option. A bidirectional GRU sweeps the profile sequentially and a GRU decoder unrolls the embedding back into the curve; the cheapest model that still models ordering explicitly.",
                    },
                    {
                        "key": "cnn_attn_ae", "name": "Conv-Attention Autoencoder", "skip": "Conv tokens + attention",
                        "head": "Tokenized transformer to embedding", "params": "~1.92M", "recommended": False,
                        "when": "The most expressive profile model. A convolutional tokenizer splits the profile into patches that a real multi-token self-attention stack relates to one another, combining local convolution with global attention at the highest parameter cost.",
                    },
                ],
            },
        ]
