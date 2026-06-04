from __future__ import annotations


class ModelLibrary:

    def collect(self) -> list[dict]:
        return [
            {
                "family" : "CNN encoder-decoder",
                "blurb"  : "Convolutional U-shaped backbones. Strong baselines, modest parameter budgets.",
                "models" : [
                    {
                        "key": "unet", "name": "UNet", "skip": "Direct concatenation",
                        "head": "Single 1x1 conv", "params": "~7M", "recommended": False,
                        "when": "The default starting point. Small datasets or limited GPU memory; minimal complexity, strong baseline.",
                    },
                    {
                        "key": "resunet", "name": "ResUNet", "skip": "Skip + residual",
                        "head": "Single 1x1 conv", "params": "~7M", "recommended": True,
                        "when": "Unstable or very deep training. Residual encoder blocks prevent gradient vanishing. Best performer in the benchmark.",
                    },
                    {
                        "key": "attention_unet", "name": "Attention UNet", "skip": "Attention-gated concat",
                        "head": "Single 1x1 conv", "params": "~8M", "recommended": False,
                        "when": "Spatially heterogeneous scenes (mixed urban and vegetated). Gates suppress irrelevant skip features per region.",
                    },
                    {
                        "key": "unetplusplus", "name": "UNet++", "skip": "Nested dense skips",
                        "head": "Single 1x1 conv", "params": "~9M", "recommended": False,
                        "when": "Skip-connection quality is the bottleneck. Graduated fusion reduces the encoder-decoder semantic gap.",
                    },
                    {
                        "key": "linknet", "name": "LinkNet", "skip": "Additive skip",
                        "head": "Single 1x1 conv", "params": "~4M", "recommended": False,
                        "when": "Efficiency is the priority. Additive skips shrink decoder channels and the parameter budget.",
                    },
                ],
            },
            {
                "family" : "Transformer",
                "blurb"  : "Global attention for long-range spatial dependencies, at a higher parameter cost.",
                "models" : [
                    {
                        "key": "swin_unet", "name": "Swin-UNet", "skip": "Hierarchical Swin",
                        "head": "1x1 conv", "params": "~28M", "recommended": False,
                        "when": "Large homogeneous structures spanning many pixels. Windowed attention scales better than full ViT.",
                    },
                    {
                        "key": "transunet", "name": "TransUNet", "skip": "Transformer patch tokens",
                        "head": "CNN decoder", "params": "~105M", "recommended": False,
                        "when": "CNN locality fused with global ViT context. Heaviest model; use when capacity is not the constraint.",
                    },
                    {
                        "key": "unetr", "name": "UNETR", "skip": "Transformer skip outputs",
                        "head": "CNN decoder", "params": "~92M", "recommended": False,
                        "when": "Pure ViT encoder with a CNN decoder. Global receptive field from the first layer.",
                    },
                ],
            },
            {
                "family" : "Multi-head CNN",
                "blurb"  : "UNet backbones with stronger inductive biases on the parameter structure.",
                "models" : [
                    {
                        "key": "unet_multihead", "name": "UNet Multihead", "skip": "Direct concatenation",
                        "head": "3 PixelMLP heads", "params": "~8M", "recommended": False,
                        "when": "Systematic bias across parameter types. Separate heads for amplitude, mean, and spread.",
                    },
                    {
                        "key": "unet_pergaussian", "name": "UNet Per-Gaussian", "skip": "Direct concatenation",
                        "head": "K PixelMLP heads", "params": "~8M", "recommended": False,
                        "when": "Systematic bias across Gaussian slots. One head per component imposes slot independence.",
                    },
                ],
            },
        ]
