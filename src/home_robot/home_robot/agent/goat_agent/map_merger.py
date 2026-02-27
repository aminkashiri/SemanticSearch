import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Optional, Tuple, Dict


class MapConstants:
    NON_SEM_CHANNELS = 10
    OBSTACLE_MAP = 0
    EXPLORED_MAP = 1
    AGENT_VISITED_MAP = 2
    VISITED_MAP = 3
    BEEN_CLOSE_MAP = 4
    BLACKLISTED_TARGETS_MAP = 5
    UNREACHABLE_FRONTIERS_MAP = 6
    GROUND_PLANE = 7
    STAIRS = 8
    GAZE_EXPLORED_MAP = 9


MC = MapConstants


class MapMerger:
    def __init__(
        self,
        num_sem_categories: int,
        resolution,
        ransac_thresh: float = 10.0,
        iou_threshold: float = 0.3,
        semantic_weight: float = 1.0,
        vis_dir: Optional[str] = None,
    ):
        self.num_sem_categories = num_sem_categories # Note that this includes last layer which is other robots loc
        self.resolution = resolution
        self.ransac_thresh = ransac_thresh
        self.iou_threshold = iou_threshold
        self.semantic_weight = semantic_weight
        self.vis_dir = vis_dir
        self.timestep = 0
        self._cached_transforms: Dict[int, np.ndarray] = {}

    def clear_cached_transforms(self):
        self._cached_transforms.clear()

    def has_cached_transform(self, neighbor_id: int) -> bool:
        return neighbor_id in self._cached_transforms

    def transform_location(self, neighbor_id: int, location) -> Optional[torch.Tensor]:
        if neighbor_id not in self._cached_transforms:
            return None

        transform = self._cached_transforms[neighbor_id]
        T = torch.from_numpy(transform).to(dtype=torch.float64)
        p = (T @ torch.tensor([location[1], location[0], 1.0], dtype=torch.float64)).round()
        return (int(p[1]), int(p[0]))

    def get_transformed_map(
        self,
        global_map: Tensor,  # (C, H, W) - this robot's map
        neighbor_global_map: Tensor,  # (C, H, W) - neighbor's map
        neighbor_id: int,
    ) -> Dict:
        """
        Align neighbor's map to this robot's frame using feature matching
        on obstacle + semantic layers, then merge.

        Returns dict with:
            - merged_map: (C, H, W) tensor
            - transform: (2, 3) np array or None if alignment failed
            - neighbor_loc_in_our_frame: (2,) int tensor or None
            - distance_cells: float or None
            - distance_meters: float or None
        """
        device = global_map.device
        if not neighbor_global_map is None:
            neighbor_global_map = neighbor_global_map.to(device)

        if self.has_cached_transform(neighbor_id):
            transform = self._cached_transforms[neighbor_id]
        elif not neighbor_global_map is None:
            transform = self._estimate_transform(global_map, neighbor_global_map)
        else:
            transform = None

        if transform is None:
            print(
                "[MapMerger] WARNING: Could not estimate transform. "
                "Returning original map unchanged."
            )
            return None

        # Warp neighbor map to our frame
        warped_neighbor = self._warp_map(
            neighbor_global_map, transform, global_map.shape
        )

        # Only check IoU for newly estimated transforms
        if neighbor_id not in self._cached_transforms:
            iou = self._alignment_confidence(global_map, warped_neighbor)
            print(f"[MapMerger] Alignment IoU: {iou:.4f} (threshold: {self.iou_threshold})")

            if iou < self.iou_threshold:
                print(
                    "[MapMerger] WARNING: Low alignment confidence. "
                    "Returning original map unchanged."
                )
                return None

            self._cached_transforms[neighbor_id] = transform
            print(f"[MapMerger] Transform cached for neighbor {neighbor_id}")

        return warped_neighbor

    def _alignment_confidence(self, map_A: Tensor, warped_B: Tensor) -> float:
        exp_A = map_A[MC.EXPLORED_MAP].cpu().numpy() > 0
        exp_B = warped_B[MC.EXPLORED_MAP].cpu().numpy() > 0
        overlap = exp_A & exp_B

        if overlap.sum() < 50:
            return 0.0

        obs_A = map_A[MC.OBSTACLE_MAP].cpu().numpy() > 0.5
        obs_B = warped_B[MC.OBSTACLE_MAP].cpu().numpy() > 0.5

        a = obs_A[overlap]
        b = obs_B[overlap]

        intersection = (a & b).sum()
        union = (a | b).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def _estimate_transform(self, map_A: Tensor, map_B: Tensor) -> Optional[np.ndarray]:
        """
        Estimate rigid transform (rotation + translation) from B's frame to A's frame
        using ORB features on obstacle map + semantic landmark correspondences.
        Returns 2x3 affine matrix or None.
        """
        obs_A = (map_A[MC.OBSTACLE_MAP].cpu().numpy() * 255).astype(np.uint8)
        obs_B = (map_B[MC.OBSTACLE_MAP].cpu().numpy() * 255).astype(np.uint8)

        # Use explored map to weight features (only match in explored regions)
        exp_A = (map_A[MC.EXPLORED_MAP].cpu().numpy() > 0).astype(np.uint8) * 255
        exp_B = (map_B[MC.EXPLORED_MAP].cpu().numpy() > 0).astype(np.uint8) * 255

        pts_A, pts_B = [], []

        # --- ORB features on obstacle map ---
        orb_pA, orb_pB = self._orb_matches(obs_A, obs_B, exp_A, exp_B)
        if orb_pA is not None:
            pts_A.append(orb_pA)
            pts_B.append(orb_pB)

        # --- Semantic landmark matching ---
        sem_pA, sem_pB = self._semantic_landmark_matches(map_A, map_B)
        if sem_pA is not None:
            pts_A.append(sem_pA)
            pts_B.append(sem_pB)

        if len(pts_A) == 0:
            print("------------ 1", len(pts_A))
            return None

        all_pts_A = np.vstack(pts_A).astype(np.float32)
        all_pts_B = np.vstack(pts_B).astype(np.float32)

        if len(all_pts_A) < 3:
            print("------------ 2", len(all_pts_A))
            return None

        # RANSAC for rigid transform (rotation + translation, no scale)
        M, inliers = cv2.estimateAffinePartial2D(
            all_pts_B.reshape(-1, 1, 2),
            all_pts_A.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
        )

        if M is None or inliers is None:
            print("------------ 3")
            return None

        return M

    def _orb_matches(
        self,
        obs_A: np.ndarray,
        obs_B: np.ndarray,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        max_features: int = 1000,
        top_k: int = 100,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        orb = cv2.ORB_create(max_features)

        kp1, des1 = orb.detectAndCompute(obs_A, mask_A)
        kp2, des2 = orb.detectAndCompute(obs_B, mask_B)

        if des1 is None or des2 is None or len(des1) < 3 or len(des2) < 3:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:top_k]

        if len(matches) < 3:
            return None, None

        pts_A = np.array([kp1[m.queryIdx].pt for m in matches])
        pts_B = np.array([kp2[m.trainIdx].pt for m in matches])
        return pts_A, pts_B

    def _semantic_landmark_matches(
        self,
        map_A: Tensor,
        map_B: Tensor,
        min_component_size: int = 10,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Match semantic object centroids between maps.
        Same semantic class + similar local obstacle context = putative match.
        """
        sem_start = MC.NON_SEM_CHANNELS
        sem_end = MC.NON_SEM_CHANNELS + self.num_sem_categories-1 #! Don't consider last layer which is other robots loc

        sem_A = map_A[sem_start:sem_end].cpu().numpy()
        sem_B = map_B[sem_start:sem_end].cpu().numpy()
        obs_A = map_A[MC.OBSTACLE_MAP].cpu().numpy()
        obs_B = map_B[MC.OBSTACLE_MAP].cpu().numpy()

        landmarks_A = self._extract_semantic_landmarks(sem_A, obs_A, min_component_size)
        landmarks_B = self._extract_semantic_landmarks(sem_B, obs_B, min_component_size)

        if not landmarks_A or not landmarks_B:
            return None, None

        pts_A, pts_B = [], []

        for la in landmarks_A:
            best_dist = float("inf")
            best_lb = None
            for lb in landmarks_B:
                if la["label"] != lb["label"]:
                    continue
                geo_dist = np.linalg.norm(la["descriptor"] - lb["descriptor"])
                area_ratio = min(la["area"], lb["area"]) / max(la["area"], lb["area"])
                score = geo_dist - self.semantic_weight * area_ratio
                if score < best_dist:
                    best_dist = score
                    best_lb = lb
            if best_lb is not None:
                pts_A.append(la["centroid"])
                pts_B.append(best_lb["centroid"])

        if len(pts_A) < 3:
            return None, None

        return np.array(pts_A), np.array(pts_B)

    def _extract_semantic_landmarks(
        self,
        sem_channels: np.ndarray,
        obstacle_map: np.ndarray,
        min_size: int,
    ) -> list:
        """Extract centroids + descriptors for each semantic object instance."""
        landmarks = []

        for cat_idx in range(self.num_sem_categories - 1):
            binary = (sem_channels[cat_idx] > 0.5).astype(np.uint8)
            if binary.sum() < min_size:
                continue

            n_comp, comp_map = cv2.connectedComponents(binary)
            for comp_id in range(1, n_comp):
                mask = comp_map == comp_id
                if mask.sum() < min_size:
                    continue

                ys, xs = np.where(mask)
                cx, cy = float(xs.mean()), float(ys.mean())
                area = len(xs)

                desc = self._radial_obstacle_descriptor(obstacle_map, cx, cy)

                landmarks.append(
                    {
                        "label": cat_idx,
                        "centroid": np.array([cx, cy]),
                        "area": area,
                        "descriptor": desc,
                    }
                )
        return landmarks

    def _radial_obstacle_descriptor(
        self,
        obs_map: np.ndarray,
        cx: float,
        cy: float,
        radius: int = 25,
        n_rings: int = 8,
    ) -> np.ndarray:
        """Rotation-invariant descriptor: obstacle density in concentric rings."""
        h, w = obs_map.shape
        y0, y1 = max(0, int(cy) - radius), min(h, int(cy) + radius)
        x0, x1 = max(0, int(cx) - radius), min(w, int(cx) + radius)
        patch = obs_map[y0:y1, x0:x1]

        ph, pw = patch.shape
        if ph < 3 or pw < 3:
            return np.zeros(n_rings)

        local_cy, local_cx = cy - y0, cx - x0
        r_max = min(local_cx, local_cy, pw - local_cx, ph - local_cy, radius)
        if r_max < 2:
            return np.zeros(n_rings)

        ys, xs = np.ogrid[:ph, :pw]
        dist = np.sqrt((xs - local_cx) ** 2 + (ys - local_cy) ** 2)

        desc = np.zeros(n_rings)
        for i in range(n_rings):
            r_in = i * r_max / n_rings
            r_out = (i + 1) * r_max / n_rings
            ring = (dist >= r_in) & (dist < r_out)
            if ring.sum() > 0:
                desc[i] = patch[ring].mean()
        return desc


    def _warp_map(
        self, neighbor_map: Tensor, transform: np.ndarray, target_shape: tuple
    ) -> Tensor:
        """Warp all channels of neighbor_map into our frame."""
        C, H, W = target_shape
        device = neighbor_map.device
        warped = torch.zeros(C, H, W, dtype=neighbor_map.dtype, device=device)

        for c in range(C):
            layer = neighbor_map[c].cpu().numpy().astype(np.float32)
            w = cv2.warpAffine(
                layer,
                transform,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            warped[c] = torch.from_numpy(w).to(device)

        return warped

    def _merge(self, global_map: Tensor, transformed_map) -> Tensor:
        """
        Merge warped neighbor into our map.
        - Protected (agent-specific) channels: untouched
        - Mergeable channels (up to semantics): max
        - Instance channels (beyond semantics): untouched
        """
        merged = global_map.clone()
        protected_channels = torch.tensor(
            [MC.AGENT_VISITED_MAP, MC.BLACKLISTED_TARGETS_MAP, MC.NON_SEM_CHANNELS + self.num_sem_categories],
            device=global_map.device,
        )
        all_channels = torch.arange(global_map.shape[0], device=global_map.device)
        merge_mask = ~torch.isin(all_channels, protected_channels) & (
            all_channels < (MC.NON_SEM_CHANNELS + self.num_sem_categories)
        )
        merged[merge_mask] = torch.maximum(
            global_map[merge_mask],
            transformed_map[merge_mask],
        )

        return merged

    @staticmethod
    def _transform_location(loc: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Transform (x, y) cell coords from neighbor's frame to our frame.
        Convention: loc = (row, col), i.e. map[x, y] = map[row, col].
        The cv2 affine transform operates in (col, row) space, so we swap.
        """
        row, col = loc[0], loc[1]
        p = transform @ np.array([col, row, 1.0])
        out_col, out_row = p[0], p[1]
        return np.round(np.array([out_row, out_col])).astype(np.int64)

    def _visualize_failed(
        self,
        map_A: Tensor,
        map_B: Tensor,
        loc_A: np.ndarray,
        loc_B: np.ndarray,
        reason=None,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        obs_A = map_A[MC.OBSTACLE_MAP].cpu().numpy()
        obs_B = map_B[MC.OBSTACLE_MAP].cpu().numpy()

        axes[0].imshow(obs_A, cmap="gray", origin="upper")
        axes[0].plot(loc_A[1], loc_A[0], "bo", markersize=10, label="Robot A")
        axes[0].set_title("Robot A - Obstacle Map")
        axes[0].legend()
        axes[0].axis("off")

        axes[1].imshow(obs_B, cmap="gray", origin="upper")
        axes[1].plot(loc_B[1], loc_B[0], "ro", markersize=10, label="Robot B")
        axes[1].set_title("Robot B - Obstacle Map (own frame)")
        axes[1].legend()
        axes[1].axis("off")

        if reason == "dist":
            plt.suptitle("Map Merge FAILED - Distance too far", color="red")
        else:
            plt.suptitle("Map Merge FAILED - Alignment could not be estimated", color="red")
        plt.tight_layout()
        path = os.path.join(self.vis_dir, f"{self.timestep}.15.map_merge_FAILED.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")

    def _visualize(self, map_A, loc_A, merged, data):
        transformed_map = data["transformed_map"]
        original_map = data["map"]  # original unwarped map
        loc_B_original = data["location"]
        loc_B_transformed = data["transformed_loc"]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        obs_A = map_A[MC.OBSTACLE_MAP].cpu().numpy()
        obs_merged = merged[MC.OBSTACLE_MAP].cpu().numpy()

        # Robot A's map
        axes[0].imshow(obs_A, cmap="gray", origin="upper")
        axes[0].plot(loc_A[1], loc_A[0], "bo", markersize=10, label="Robot A")
        axes[0].set_title("Robot A - Obstacle Map")
        axes[0].legend()
        axes[0].axis("off")

        obs_B_original = original_map[MC.OBSTACLE_MAP].cpu().numpy()
        axes[1].imshow(obs_B_original, cmap="gray", origin="upper")
        axes[1].plot(loc_B_original[1], loc_B_original[0], "ro", markersize=10, label="Robot B")
        axes[1].set_title("Robot B - Obstacle Map (own frame)")
        axes[1].legend()
        axes[1].axis("off")

        # Panel 3: Overlay — use already-warped map directly, no second warp
        obs_B_warped = transformed_map[MC.OBSTACLE_MAP].cpu().numpy()
        overlay = np.zeros((*obs_A.shape, 3))
        overlay[:, :, 2] = np.clip(obs_A, 0, 1)
        overlay[:, :, 0] = np.clip(obs_B_warped, 0, 1)
        axes[2].imshow(overlay, origin="upper")
        axes[2].plot(loc_A[1], loc_A[0], "bo", markersize=10, label="Robot A")
        axes[2].plot(loc_B_transformed[1], loc_B_transformed[0], "ro", markersize=10, label="Robot B (aligned)")
        axes[2].set_title("Alignment Overlay (blue=A, red=B)")
        axes[2].legend()
        axes[2].axis("off")

        # Panel 4: Merged map
        axes[3].imshow(obs_merged, cmap="gray", origin="upper")
        axes[3].plot(loc_A[1], loc_A[0], "bo", markersize=10, label="Robot A")
        axes[3].plot(loc_B_transformed[1], loc_B_transformed[0], "ro", markersize=10, label="Robot B")
        dist_cells = np.sqrt(
            (loc_A[0] - loc_B_transformed[0]) ** 2
            + (loc_A[1] - loc_B_transformed[1]) ** 2
        )
        axes[3].set_title(f"Merged Map (dist={dist_cells * self.resolution:.2f}m)")
        axes[3].legend()
        axes[3].axis("off")

        plt.tight_layout()
        path = os.path.join(self.vis_dir, f"{self.timestep}.15.map_merge_result.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")



if __name__ == "__main__":
    map_A = torch.load("/home/agilex2/projects/search/SemanticSearch/datadump/images/comm3/real_world_0/agent_1/mymap_40.pt")
    map_B = torch.load("/home/agilex2/projects/search/SemanticSearch/datadump/images/comm3/real_world_0/agent_1/othermap_40.pt")
    num_sem = int((map_A.shape[0] - MC.NON_SEM_CHANNELS) / 2)
    print(f"Map channels: {map_A.shape[0]}, num_sem: {num_sem}")

    # Robot locations in their own frames (x, y) in pixel coords
    loc_A = [502, 538]
    loc_B = [434, 406]
    data = {
        "agent_id": 1,
        "map": map_B,
        "location": loc_B
    }

    map_merger = MapMerger(
        num_sem_categories=num_sem,
        resolution=0.05,
        ransac_thresh=10.0,
        iou_threshold=0.2,
    )
    map_merger.vis_dir = "."


    transfomed_map = map_merger.get_transformed_map(
        global_map=map_A,
        neighbor_global_map=map_B,
        neighbor_id=1
    )
    merged = map_merger._merge(
        map_A,
        transfomed_map,
    )

    transformed_loc = map_merger.transform_location(
        data["agent_id"], data["location"]
    )
    data["transformed_map"] = transfomed_map
    data["transformed_loc"] = transformed_loc

    map_merger._visualize(
        map_A,
        loc_A,
        merged,
        data,
    )
