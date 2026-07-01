"""
finetune_config.py  —  FR3 joint-position finetuning config for Octo

Usage:
    python scripts/finetune.py \
        --config experiments/fr3/finetune_config.py:full,language_conditioned \
        --config.pretrained_path hf://rail-berkeley/octo-base-1.5 \
        --config.save_dir /path/to/save/checkpoints \
        --config.wandb.entity your_wandb_entity \
        --config.wandb.group fr3_finetune_run1

Finetuning modes (first arg):
    full            — finetune everything (recommended for FR3, new action space)
    head_mlp_only   — freeze backbone, train MLP head only (faster, less data needed)
    head_only       — freeze everything except readout head (minimal compute)

Task modality (second arg):
    language_conditioned  — robot receives text instruction only
    image_conditioned     — robot receives goal image only
    multimodal            — randomly uses either (requires goal image in dataset)
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # ── dataset config ────────────────────────────────────────────────────────
    FINETUNING_KWARGS = {
        # Name must match the folder/class name of your TFDS builder
        "name": "fr3_demo_dataset",

        # Path where tfds build wrote the TFRecord shards
        "data_dir": "/path/to/tfds_output",

        # Match the image keys defined in fr3_dataset_builder.py
        "image_obs_keys": {
            "primary": "image_primary",  # 3rd-person camera (256x256)
            "wrist":   "image_wrist",    # wrist camera (128x128)
        },

        # "proprio" is set by fr3_standardize_fn — 8-dim: 7 joints + gripper
        "proprio_obs_key": "proprio",

        # Language instruction key (set in fr3_standardize_fn under task)
        "language_key": "language_instruction",

        # Normalize actions — True means normalize that dimension.
        # 8 dimensions total: 7 joint positions (normalize) + 1 gripper (don't normalize)
        # Gripper is already in a meaningful [0,1] range so skip normalization.
        "action_normalization_mask": [True, True, True, True, True, True, True, False],

        # Action normalization strategy
        "action_proprio_normalization_type": "normal",

        # Your custom standardize function
        "standardize_fn": ModuleSpec.create(
            "fr3_standardize_fn:fr3_dataset_transform"
        ),
    }

    # ── frozen keys per mode ──────────────────────────────────────────────────
    if mode == "full":
        # Recommended for FR3: new action space needs full adaptation
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )

    max_steps  = FieldReference(50000)
    window_size = FieldReference(default=2)  # use 2 to leverage Octo's history window

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),

        # ── training hyperparameters ──────────────────────────────────────────
        # Reduce batch_size if you run out of GPU memory (try 64 or 128)
        batch_size=256,
        shuffle_buffer_size=10000,
        num_steps=max_steps,

        # ── logging / saving ──────────────────────────────────────────────────
        log_interval=100,
        eval_interval=2000,   # evaluate on val set every N steps
        save_interval=2000,   # save checkpoint every N steps
        save_dir=placeholder(str),
        seed=42,

        wandb=dict(
            project="octo_fr3_finetune",
            group=placeholder(str),
            entity=placeholder(str),
        ),

        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,

        # ── optimizer ─────────────────────────────────────────────────────────
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,     # lower to 1e-4 if training is unstable
                warmup_steps=1000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),

        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),

        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    # ── task-specific data pipeline settings ──────────────────────────────────
    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5

    # ── trajectory transform ──────────────────────────────────────────────────
    traj_transform_kwargs = dict(
        window_size=window_size,

        # How many future actions Octo predicts per inference call.
        # At ~10Hz control, action_horizon=4 gives 0.4s of predicted motion
        # before the next Octo call. Increase if inference is slow.
        action_horizon=4,

        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )

    # ── frame (image) transform ───────────────────────────────────────────────
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        # No crop on wrist — framing is important for close-up manipulation
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),
            "wrist":   (128, 128),
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            wrist=wrist_augment_kwargs,
        ),
    )

    config["frame_transform_threads"]  = 16
    config["traj_transform_kwargs"]    = traj_transform_kwargs
    config["frame_transform_kwargs"]   = frame_transform_kwargs

    return ConfigDict(config)
