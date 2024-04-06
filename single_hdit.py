import k_diffusion as K
from k_diffusion import models


def model_from_config(config, num_classes=0):
    assert len(config["widths"]) == len(config["depths"])
    assert len(config["widths"]) == len(config["d_ffs"])
    assert len(config["widths"]) == len(config["self_attns"])
    assert len(config["widths"]) == len(config["dropout_rate"])
    levels = []

    for depth, width, d_ff, self_attn, dropout in zip(
        config["depths"],
        config["widths"],
        config["d_ffs"],
        config["self_attns"],
        config["dropout_rate"],
    ):
        if self_attn["type"] == "global":
            self_attn = models.image_transformer_v2.GlobalAttentionSpec(
                self_attn.get("d_head", 64)
            )
        elif self_attn["type"] == "neighborhood":
            self_attn = models.image_transformer_v2.NeighborhoodAttentionSpec(
                self_attn.get("d_head", 64), self_attn.get("kernel_size", 7)
            )
        elif self_attn["type"] == "shifted-window":
            self_attn = models.image_transformer_v2.ShiftedWindowAttentionSpec(
                self_attn.get("d_head", 64), self_attn["window_size"]
            )
        elif self_attn["type"] == "none":
            self_attn = models.image_transformer_v2.NoAttentionSpec()
        else:
            raise ValueError(f'unsupported self attention type {self_attn["type"]}')
        levels.append(
            models.image_transformer_v2.LevelSpec(
                depth, width, d_ff, self_attn, dropout
            )
        )
    mapping = models.image_transformer_v2.MappingSpec(
        config["mapping_depth"],
        config["mapping_width"],
        config["mapping_d_ff"],
        config["mapping_dropout_rate"],
    )
    model = models.ImageTransformerDenoiserModelV2(
        levels=levels,
        mapping=mapping,
        in_channels=config["input_channels"],
        out_channels=config["input_channels"],
        patch_size=config["patch_size"],
        num_classes=num_classes + 1 if num_classes else 0,
        mapping_cond_dim=config["mapping_cond_dim"],
    )
    return model


_config = K.config.load_config(
    {
        "model": {
            "type": "image_transformer_v2",
            "input_channels": 3,
            "input_size": [256, 256],
            "patch_size": [4, 4],
            "depths": [2, 2, 16],
            "widths": [384, 768, 1536],
            "loss_config": "karras",
            "loss_weighting": "soft-min-snr",
            "loss_scales": 1,
            "dropout_rate": [0.0, 0.0, 0.0],
            "mapping_dropout_rate": 0.0,
            "augment_prob": 0.0,
            "sigma_data": 0.5,
            "sigma_min": 1e-2,
            "sigma_max": 160,
            "sigma_sample_density": {"type": "cosine-interpolated"},
        },
        "dataset": {
            "cond_dropout_rate": 0.1,
            "num_classes": 1000,
        },
    }
)
model = K.config.make_model(_config)

print(model)
print(sum(p.numel() for p in model.parameters()))
