# Octo Finetuning Pipeline — FR3 + ROS2 Humble

## Files

| File | Purpose |
|---|---|
| `extract_bags.py` | Step 1 — rosbags → HDF5 episodes |
| `fr3_dataset_builder.py` | Step 2 — HDF5 → RLDS/TFDS format |
| `fr3_standardize_fn.py` | Step 3 — maps your keys to Octo's internal keys |
| `finetune_config.py` | Step 4 — Octo finetuning config |

---

## Topics used from your rosbags

| Topic | Used as |
|---|---|
| `/camera/primary/image_raw` | Primary (3rd-person) observation image |
| `/camera/wrist/image_raw` | Wrist observation image |
| `/franka_robot_state_broadcaster/measured_joint_states` | Arm joint positions (observation) |
| `/franka_gripper/joint_states` | Gripper width (observation) |
| `/franka_robot_state_broadcaster/desired_joint_states` | Arm joint command (action) |
| `/gripper/gripper_client/target_gripper_width_percent` | Gripper command (action) |

---

## Step-by-step

### Step 1 — Extract rosbags to HDF5

```bash
python extract_bags.py \
    --bag_dir /path/to/your/rosbag/folders \
    --output_dir /path/to/hdf5_episodes \
    --language_instruction "pick up the cube and place it in the bin"
```

Each rosbag folder → one `episode_NNNN.hdf5` file.

Structure inside each HDF5:
```
episode_0001.hdf5
├── observations/
│   ├── joint_positions   (T, 7)
│   ├── gripper_state     (T, 1)
│   └── images/
│       ├── primary       (T, 256, 256, 3)
│       └── wrist         (T, 128, 128, 3)
├── actions               (T, 8)   ← 7 joint commands + gripper
└── attrs: language_instruction
```

**Verify your extraction looks right before proceeding:**
```python
import h5py, numpy as np
with h5py.File("episode_0000.hdf5", "r") as f:
    print("Steps:  ", f["actions"].shape[0])
    print("Actions:", f["actions"][:3])           # first 3 action vectors
    print("Lang:   ", f.attrs["language_instruction"])
```

---

### Step 2 — Build the TFDS dataset

Set the path to your HDF5 files:
```bash
export HDF5_EPISODE_DIR=/path/to/hdf5_episodes
```

Place `fr3_dataset_builder.py` inside a folder called `fr3_demo_dataset/`:
```
fr3_demo_dataset/
    __init__.py              # empty file
    fr3_dataset_builder.py
```

Build it:
```bash
cd /parent/of/fr3_demo_dataset
tfds build fr3_demo_dataset --data_dir=/path/to/tfds_output
```

This produces sharded TFRecord files:
```
/path/to/tfds_output/fr3_demo_dataset/1.0.0/
    fr3_demo_dataset-train.tfrecord-00000-of-NNNNN
    ...
    fr3_demo_dataset-val.tfrecord-00000-of-NNNNN
    dataset_info.json
```

---

### Step 3 — Run finetuning

Place `fr3_standardize_fn.py` somewhere importable (e.g. in the octo repo root or on your PYTHONPATH).

Update the two paths in `finetune_config.py`:
- `data_dir` → `/path/to/tfds_output`
- `save_dir` → where you want checkpoints saved

Then run:
```bash
python scripts/finetune.py \
    --config experiments/fr3/finetune_config.py:full,language_conditioned \
    --config.pretrained_path hf://rail-berkeley/octo-base-1.5 \
    --config.save_dir /path/to/checkpoints \
    --config.wandb.entity your_wandb_entity \
    --config.wandb.group fr3_run1
```

---

## Action space summary

| Dimension | Content | Normalized? |
|---|---|---|
| 0–6 | Desired joint positions (7 DOF) | Yes |
| 7 | Gripper command (0–1) | No |

## Tips

- **Not enough data?** Start with `head_mlp_only` mode instead of `full` — needs less data
- **Inference too slow?** Increase `action_horizon` in the config so each Octo call covers more steps before re-querying
- **Sync issues in bags?** Check timestamps with `ros2 bag info` — if your camera ran at a different rate than joint states, the nearest-neighbor sync in `extract_bags.py` handles it but large gaps (>100ms) will show as jitter
- **Multiple tasks?** Run `extract_bags.py` separately per task with different `--language_instruction`, then combine all HDF5 files into one directory before building the TFDS dataset
