from __future__ import annotations

import re
import sys
from pathlib import Path


class ImageAutoencoderModelLibrary:

    OPERATIONAL_PAREN   = re.compile(r"\s*\([^()]*(?:\d{4}-\d{2}-\d{2}|referee|first[- ]pass|regression|corrected|reviewed|audit)[^()]*\)", re.IGNORECASE)
    OPERATIONAL_MARKERS = re.compile(r"\d{4}-\d{2}-\d{2}|referee|first[- ]pass|smoke[- ]test|regression|review date|reviewed|corrected on|correction|audit|flagged", re.IGNORECASE)
    OPERATIONAL_HEADING = re.compile(r"correction|checkpoint continuity|review", re.IGNORECASE)

    CONFIG_CLASSES = {
        "conv2d_ae"     : "Conv2dImageAutoencoderConfig",
        "resnet2d_ae"   : "ResNet2dImageAutoencoderConfig",
        "convnext2d_ae" : "ConvNeXt2dImageAutoencoderConfig",
        "dilated2d_ae"  : "DilatedConv2dImageAutoencoderConfig",
        "vit_ae"        : "ViTImageAutoencoderConfig",
    }

    NOTE_FILES = {
        "conv2d_ae"     : "Conv2D Image Autoencoder.md",
        "resnet2d_ae"   : "ResNet2D Image Autoencoder.md",
        "convnext2d_ae" : "ConvNeXt2D Image Autoencoder.md",
        "dilated2d_ae"  : "Dilated2D Image Autoencoder.md",
        "vit_ae"        : "ViT Image Autoencoder.md",
    }

    FALLBACK_ACTIVATION    = "gelu"
    FALLBACK_NORMALIZATION = "batch"

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
        module   = self._import_image_autoencoder_models_config()
        resolved = {}

        for key, class_name in self.CONFIG_CLASSES.items():
            config        = getattr(module, class_name)()
            activation    = getattr(config, "activation", self.FALLBACK_ACTIVATION)
            normalization = getattr(config, "normalization", self.FALLBACK_NORMALIZATION)
            resolved[key] = (str(activation).lower(), str(normalization).lower())

        return resolved

    def _import_image_autoencoder_models_config(self):
        config_dir = Path(__file__).resolve().parent.parent / "configuration" / "architectures"
        path       = str(config_dir)

        if path not in sys.path:
            sys.path.insert(0, path)

        import image_autoencoder

        return image_autoencoder

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "Image autoencoder",
                "blurb"  : "Compress the SAR image stack into a 2D spatial embedding and reconstruct it. The encoder later serves as the JEPA image front-end.",
                "models" : [
                    {
                        "key": "conv2d_ae", "name": "Conv2D Image Autoencoder", "skip": "2D conv encoder/decoder",
                        "head": "Conv to 2D embedding", "params": "~48.4K", "recommended": True,
                        "when": "The default image autoencoder. A small 2D convolutional encoder downsamples the SAR image stack to a spatial embedding and a mirrored decoder reconstructs it; the encoder is reused as the pretrained JEPA image front-end.",
                    },
                    {
                        "key": "resnet2d_ae", "name": "ResNet2D Image Autoencoder", "skip": "Residual 2D conv",
                        "head": "Residual conv to 2D embedding", "params": "~208.2K", "recommended": False,
                        "when": "A deeper, residual alternative to the default. Pre-activation residual blocks with strided downsampling give a more expressive encoder that trains stably at greater depth, at a higher parameter cost.",
                    },
                    {
                        "key": "convnext2d_ae", "name": "ConvNeXt2D Image Autoencoder", "skip": "ConvNeXt blocks",
                        "head": "ConvNeXt to 2D embedding", "params": "~143.3K", "recommended": False,
                        "when": "A modern convolutional design. Depthwise 7x7 convolutions, layer normalisation and inverted bottlenecks capture wider spatial context than plain 3x3 stacks while staying parameter-efficient.",
                    },
                    {
                        "key": "dilated2d_ae", "name": "Dilated2D Image Autoencoder", "skip": "Atrous residual conv",
                        "head": "Dilated conv to 2D embedding", "params": "~113.3K", "recommended": False,
                        "when": "Preserves full spatial resolution. Stacked dilated residual convolutions grow the receptive field without downsampling, so the embedding keeps the input grid size; best when fine spatial detail must survive the bottleneck.",
                    },
                    {
                        "key": "vit_ae", "name": "ViT Image Autoencoder", "skip": "Patch attention",
                        "head": "Transformer to 2D embedding", "params": "~3.6M", "recommended": False,
                        "when": "The most expressive image model. Patches are tokenised and related by a transformer encoder and decoder with convolutional position encoding, modelling global structure across the stack at the highest parameter cost.",
                    },
                ],
            },
        ]
