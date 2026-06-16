from __future__ import annotations

import re
import sys
from pathlib import Path


class BackboneModelLibrary:

    OPERATIONAL_PAREN   = re.compile(r"\s*\([^()]*(?:\d{4}-\d{2}-\d{2}|referee|first[- ]pass|regression|corrected|reviewed|audit)[^()]*\)", re.IGNORECASE)
    OPERATIONAL_MARKERS = re.compile(r"\d{4}-\d{2}-\d{2}|referee|first[- ]pass|smoke[- ]test|regression|review date|reviewed|corrected on|correction|audit|flagged", re.IGNORECASE)
    OPERATIONAL_HEADING = re.compile(r"correction|checkpoint continuity|review", re.IGNORECASE)

    CONFIG_CLASSES = {
        "unet"               : "UNetConfig",
        "unet_skip"          : "UNetSkipConfig",
        "resunet"            : "ResUNetConfig",
        "attention_unet"     : "AttentionUNetConfig",
        "unetplusplus"       : "UNetPlusPlusConfig",
        "linknet"            : "LinkNetConfig",
        "swin_unet"          : "SwinUNetConfig",
        "transunet"          : "TransUNetConfig",
        "unetr"              : "UNETRConfig",
        "unet_multihead"     : "UNetMultiHeadConfig",
        "unet_pergaussian"   : "UNetPerGaussianConfig",
        "resunet_multihead"  : "ResUNetMultiHeadConfig",
        "resunet_pergaussian": "ResUNetPerGaussianConfig",
        "deeplabv3plus"      : "DeepLabV3PlusConfig",
        "segformer"          : "SegFormerLiteConfig",
        "convnext_unet"      : "ConvNeXtUNetConfig",
        "dense_unet"         : "DenseUNetConfig",
        "hrnet"              : "HRNetLiteConfig",
        "multires_unet"      : "MultiResUNetConfig",
        "fpn"                : "FPNNetConfig",
        "u2net"              : "U2NetLiteConfig",
    }

    NOTE_FILES = {
        "unet"               : "UNet.md",
        "unet_skip"          : "UNet Skip.md",
        "resunet"            : "ResUNet.md",
        "attention_unet"     : "Attention UNet.md",
        "unetplusplus"       : "UNet++.md",
        "linknet"            : "LinkNet.md",
        "swin_unet"          : "SwinUNet.md",
        "transunet"          : "TransUNet.md",
        "unetr"              : "UNETR.md",
        "unet_multihead"     : "UNet Multihead.md",
        "unet_pergaussian"   : "UNet Per-Gaussian.md",
        "resunet_multihead"  : "ResUNet Multihead.md",
        "resunet_pergaussian": "ResUNet Per-Gaussian.md",
        "deeplabv3plus"      : "DeepLabV3+.md",
        "segformer"          : "SegFormer.md",
        "convnext_unet"      : "ConvNeXt UNet.md",
        "dense_unet"         : "DenseUNet.md",
        "hrnet"              : "HRNet.md",
        "multires_unet"      : "MultiResUNet.md",
        "fpn"                : "FPN.md",
        "u2net"              : "U2-Net.md",
    }

    FALLBACK_ACTIVATION    = "gelu"
    FALLBACK_NORMALIZATION = "layer"

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
        module   = self._import_backbone_models_config()
        resolved = {}

        for key, class_name in self.CONFIG_CLASSES.items():
            config        = getattr(module, class_name)()
            activation    = getattr(config, "activation", self.FALLBACK_ACTIVATION)
            normalization = getattr(config, "normalization", self.FALLBACK_NORMALIZATION)
            resolved[key]      = (str(activation).lower(), str(normalization).lower())

        return resolved

    def _import_backbone_models_config(self):
        config_dir = Path(__file__).resolve().parent.parent / "configuration" / "architectures"
        path       = str(config_dir)

        if path not in sys.path:
            sys.path.insert(0, path)

        import backbone

        return backbone

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "CNN encoder-decoder",
                "blurb"  : "Convolutional U-shaped backbones. Strong baselines, modest parameter budgets.",
                "models" : [
                    {
                        "key": "unet", "name": "UNet", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~31.0M", "recommended": False,
                        "when": "The default starting point. Small datasets or limited GPU memory; minimal complexity, strong baseline.",
                    },
                    {
                        "key": "resunet", "name": "ResUNet", "skip": "Skip + residual",
                        "head": "Single 1x1 conv", "params": "~32.4M", "recommended": False,
                        "when": "Unstable or very deep training. Residual encoder blocks prevent gradient vanishing; stride-2 downsampling per Zhang et al. 2018.",
                    },
                    {
                        "key": "unet_skip", "name": "UNet Skip", "skip": "Skip + residual",
                        "head": "Single 1x1 conv", "params": "~32.4M", "recommended": True,
                        "when": "Continuity with results trained before 2026-06-04. Pre-correction ResUNet (residual blocks, MaxPool downsampling); the archived checkpoint and benchmark-winning results correspond to this architecture.",
                    },
                    {
                        "key": "attention_unet", "name": "Attention UNet", "skip": "Attention-gated concat",
                        "head": "Single 1x1 conv", "params": "~32.4M", "recommended": False,
                        "when": "Spatially heterogeneous scenes (mixed urban and vegetated). Gates suppress irrelevant skip features per region.",
                    },
                    {
                        "key": "unetplusplus", "name": "UNet++", "skip": "Nested dense skips",
                        "head": "Single 1x1 conv", "params": "~31.1M", "recommended": False,
                        "when": "Skip-connection quality is the bottleneck. Graduated fusion reduces the encoder-decoder semantic gap.",
                    },
                    {
                        "key": "linknet", "name": "LinkNet", "skip": "Additive skip",
                        "head": "Single 1x1 conv", "params": "~31.0M", "recommended": False,
                        "when": "Efficiency is the priority. Additive skips shrink decoder channels and the parameter budget.",
                    },
                    {
                        "key": "convnext_unet", "name": "ConvNeXt UNet", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~19.1M", "recommended": False,
                        "when": "Modern convolution design under the unchanged U topology. Large depthwise kernels, inverted bottlenecks, LayerNorm and GELU.",
                    },
                    {
                        "key": "dense_unet", "name": "DenseUNet", "skip": "Dense concat",
                        "head": "Single 1x1 conv", "params": "~1.0M", "recommended": False,
                        "when": "Parameter efficiency through feature reuse. At matched capacity the deepest, thinnest model in the zoo.",
                    },
                    {
                        "key": "multires_unet", "name": "MultiResUNet", "skip": "ResPath concat",
                        "head": "Single 1x1 conv", "params": "~14.4M", "recommended": False,
                        "when": "Structures span several spatial scales. Every block sees 3/5/7-pixel receptive fields in parallel, and ResPaths close the semantic gap.",
                    },
                    {
                        "key": "u2net", "name": "U2-Net", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~8.3M", "recommended": False,
                        "when": "Intra-stage multi-scale mixing is desired. Each stage is itself a small U-Net, so even full-resolution layers integrate wide context.",
                    },
                ],
            },
            {
                "family" : "Transformer",
                "blurb"  : "Global attention for long-range spatial dependencies, at a higher parameter cost.",
                "models" : [
                    {
                        "key": "swin_unet", "name": "Swin-UNet", "skip": "Hierarchical Swin",
                        "head": "1x1 conv", "params": "~28.8M", "recommended": False,
                        "when": "Large homogeneous structures spanning many pixels. Windowed attention scales better than full ViT.",
                    },
                    {
                        "key": "transunet", "name": "TransUNet", "skip": "Transformer patch tokens",
                        "head": "CNN decoder", "params": "~30.6M", "recommended": False,
                        "when": "CNN locality fused with global ViT context. Hybrid encoder; use when both fine detail and long-range structure matter.",
                    },
                    {
                        "key": "unetr", "name": "UNETR", "skip": "Transformer skip outputs",
                        "head": "CNN decoder", "params": "~34.4M", "recommended": False,
                        "when": "Pure ViT encoder with a CNN decoder. Global receptive field from the first layer.",
                    },
                    {
                        "key": "segformer", "name": "SegFormer", "skip": "Pyramid to MLP decoder",
                        "head": "1x1 conv", "params": "~5.2M", "recommended": False,
                        "when": "A strong hierarchical attention encoder with a near-trivial MLP decoder. The complementary hypothesis to the heavy-decoder transformers.",
                    },
                ],
            },
            {
                "family" : "Multi-head CNN",
                "blurb"  : "UNet backbones with stronger inductive biases on the parameter structure.",
                "models" : [
                    {
                        "key": "unet_multihead", "name": "UNet Multihead", "skip": "Direct concatenation",
                        "head": "3 PixelMLP heads", "params": "~31.0M", "recommended": False,
                        "when": "Systematic bias across parameter types. Separate heads for amplitude, mean, and spread.",
                    },
                    {
                        "key": "unet_pergaussian", "name": "UNet Per-Gaussian", "skip": "Direct concatenation",
                        "head": "K PixelMLP heads", "params": "~31.0M", "recommended": False,
                        "when": "Systematic bias across Gaussian slots. One head per component imposes slot independence.",
                    },
                    {
                        "key": "resunet_multihead", "name": "ResUNet Multihead", "skip": "Skip + residual",
                        "head": "3 PixelMLP heads", "params": "~32.4M", "recommended": False,
                        "when": "Corrected stride-2 ResUNet backbone with parameter-type-specific heads. Isolates the head-structure question from the backbone.",
                    },
                    {
                        "key": "resunet_pergaussian", "name": "ResUNet Per-Gaussian", "skip": "Skip + residual",
                        "head": "K PixelMLP heads", "params": "~32.4M", "recommended": False,
                        "when": "Corrected stride-2 ResUNet backbone with per-slot heads. Isolates the head-structure question from the backbone.",
                    },
                ],
            },
            {
                "family" : "Context and resolution",
                "blurb"  : "Dense-prediction designs that manage context and resolution differently from the symmetric U.",
                "models" : [
                    {
                        "key": "deeplabv3plus", "name": "DeepLabV3+", "skip": "Low-level fusion",
                        "head": "ASPP + decoder", "params": "~10.3M", "recommended": False,
                        "when": "Multi-scale context at fixed dilation rates matters. ASPP aggregates several receptive fields in parallel without extra downsampling.",
                    },
                    {
                        "key": "hrnet", "name": "HRNet", "skip": "Branch fusion",
                        "head": "Concat + 1x1 conv", "params": "~3.1M", "recommended": False,
                        "when": "Per-pixel position accuracy is the priority. The full-resolution stream is never downsampled, avoiding the encode-decode round trip.",
                    },
                    {
                        "key": "fpn", "name": "FPN", "skip": "Lateral additions",
                        "head": "Pyramid sum + 1x1 conv", "params": "~7.8M", "recommended": False,
                        "when": "Probing how much decoder capacity the task needs. The minimal decoder concentrates parameters in the encoder.",
                    },
                ],
            },
        ]
