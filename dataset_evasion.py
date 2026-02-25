# datasets/mantra_json_dataset.py
import os
import glob
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MantraJsonDataset3D(Dataset):
    """
    Trajectory-only MANTRA dataset that reads your JSON format (list of dicts).
    Includes noise injection logic synced with lazy_base_dataset tiers.
    """

    def __init__(
        self,
        data_path: str,
        past_len: int,
        future_len: int,
        step_size: int = 1,
        use_ego_frame: bool = True,
        yaw_key: str = "psi_rad",
        x_key: str = "x",
        y_key: str = "y",
        z_key: str = "z",
        return_dummy_scene: bool = True,
        dummy_scene_hw: int = 64,
        dummy_scene_ch: int = 4,
        max_files: Optional[int] = None,
    ):
        self.data_path = data_path
        self.past_len = int(past_len)
        self.future_len = int(future_len)
        self.total_len = self.past_len + self.future_len
        self.step_size = int(step_size)
        self.use_ego_frame = bool(use_ego_frame)

        self.yaw_key = yaw_key
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key

        self.return_dummy_scene = bool(return_dummy_scene)
        self.dummy_scene_hw = int(dummy_scene_hw)
        self.dummy_scene_ch = int(dummy_scene_ch)

        # Noise configuration (set this manually after instantiation)
        #self.noise_tier: Optional[str] = None # 'easy', 'medium', 'hard'
        self.noise_tier: str = 'hard'

        self.json_files: List[str] = glob.glob(
            os.path.join(self.data_path, "**", "*.json"),
            recursive=True,
        )
        if max_files is not None:
            self.json_files = self.json_files[: int(max_files)]

        if len(self.json_files) == 0:
            raise RuntimeError(f"No .json files found under: {self.data_path}")

        self.index_map: List[Tuple[int, int]] = []
        self._build_index_map()

        print(f"[MANTRA-3D] Initialized with {len(self.index_map)} segments across {len(self.json_files)} files")

    def _build_index_map(self):
        for file_idx, fp in enumerate(self.json_files):
            with open(fp, "r") as f:
                data = json.load(f)
            T = len(data)
            if T < self.total_len:
                continue
            for end_idx in range(self.total_len, T + 1, self.step_size):
                self.index_map.append((file_idx, end_idx))

    def _apply_robust_noise(self, traj: np.ndarray) -> np.ndarray:
        """Applies corruption logic based on lazy_base_dataset tiers."""
        if self.noise_tier is None:
            return traj

        tier = self.noise_tier.lower()
        T, D = traj.shape
        out = traj.copy()

        # 1. Tier Parameters (Synced with lazy_base_dataset.py)
        if tier == "easy":
            meas_std, rw_step, spike_p, spike_std = 0.30, 0.02, 0.002, 2.0
        elif tier == "medium":
            meas_std, rw_step, spike_p, spike_std = 1.0, 0.08, 0.01, 8.0
        elif tier == "hard":
            meas_std, rw_step, spike_p, spike_std = 2.5, 0.20, 0.03, 20.0
        else:
            return traj

        # 2. Gaussian Measurement Noise
        out += np.random.normal(0.0, meas_std, size=out.shape).astype(np.float32)

        # 3. Random Walk (Drift)
        steps = np.random.normal(0.0, rw_step, size=out.shape).astype(np.float32)
        out += np.cumsum(steps, axis=0)

        # 4. Outlier Spikes
        spike_mask = np.random.rand(T) < spike_p
        spikes = np.random.normal(0.0, spike_std, size=(T, D)).astype(np.float32)
        out[spike_mask] += spikes[spike_mask]

        return out

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, end_idx = self.index_map[idx]
        fp = self.json_files[file_idx]

        with open(fp, "r") as f:
            data = json.load(f)

        window = data[end_idx - self.total_len : end_idx]
        past = window[: self.past_len]
        future = window[self.past_len :]

        px = np.asarray([r[self.x_key] for r in past], dtype=np.float32)
        py = np.asarray([r[self.y_key] for r in past], dtype=np.float32)
        pz = np.asarray([r[self.z_key] for r in past], dtype=np.float32)

        fx = np.asarray([r[self.x_key] for r in future], dtype=np.float32)
        fy = np.asarray([r[self.y_key] for r in future], dtype=np.float32)
        fz = np.asarray([r[self.z_key] for r in future], dtype=np.float32)

        if self.use_ego_frame:
            x0, y0, z0 = px[-1], py[-1], pz[-1]
            px, py, pz = px - x0, py - y0, pz - z0
            fx, fy, fz = fx - x0, fy - y0, fz - z0

            yaw0 = float(past[-1].get(self.yaw_key, 0.0))
            c = np.cos(-yaw0).astype(np.float32)
            s = np.sin(-yaw0).astype(np.float32)

            px_r, py_r = c * px - s * py, s * px + c * py
            fx_r, fy_r = c * fx - s * fy, s * fx + c * fy
            px, py, fx, fy = px_r, py_r, fx_r, fy_r

        past_traj = np.stack([px, py, pz], axis=-1)
        
        # --- NOISE INJECTION ---
        past_traj = self._apply_robust_noise(past_traj)
        
        fut_traj = np.stack([fx, fy, fz], axis=-1)

        sample: Dict[str, Any] = {
            "index": idx,
            "past": torch.from_numpy(past_traj), 
            "future": torch.from_numpy(fut_traj), 
            "angle": torch.tensor([float(past[-1].get(self.yaw_key, 0.0))], dtype=torch.float32),
        }

        if self.return_dummy_scene:
            sample["scene"] = torch.zeros((self.dummy_scene_hw, self.dummy_scene_hw), dtype=torch.uint8)
            sample["scene_one_hot"] = torch.zeros(
                (self.dummy_scene_hw, self.dummy_scene_hw, self.dummy_scene_ch),
                dtype=torch.float32
            )

        return sample


def mantra_collate_3d(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
      past: (B,Tp,3)
      future: (B,Tf,3)
      angle: (B,1)
      optional dummy scenes
    """
    past = torch.stack([b["past"] for b in batch], dim=0)
    future = torch.stack([b["future"] for b in batch], dim=0)
    angle = torch.stack([b["angle"] for b in batch], dim=0)

    out: Dict[str, Any] = {"past": past, "future": future, "angle": angle}

    if "scene_one_hot" in batch[0]:
        out["scene_one_hot"] = torch.stack([b["scene_one_hot"] for b in batch], dim=0)
    if "scene" in batch[0]:
        out["scene"] = torch.stack([b["scene"] for b in batch], dim=0)

    return out
