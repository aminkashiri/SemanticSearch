# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import clip
import matplotlib
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from home_robot.agent.imagenav_agent.superglue import Matching
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

matplotlib.use("Agg")
MIN_PIXELS = 1000
MIN_EDGE = 15

from home_robot.utils.logger import get_logger

logger = get_logger()


class GoatMatching(Matching):
    def __init__(
        self,
        device: int,
        config: Dict[str, Any],
        default_vis_dir: str,
        print_images: bool,
        instance_memory: InstanceMemory,
        logger,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

        self.score_func = config.score_function
        assert self.score_func in ["confidence_sum", "match_count"]

        # generate clip embeddings by loading clip model
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device)
        self.instance_memory = instance_memory
        self.step = 0
        self.score_thresh = {
            "languagenav": config.score_thresh_lang,
            "imagenav": config.score_thresh_image,
            "objectnav": 0.0
        }
        self.log = logger 

    def get_matches_against_current_frame(
        self,
        task,
        use_full_image=False,
        global_pose=None,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        detected in the current frame.
        """
        # TODO We should restrict detections in the current frame by category
        detections = []
        instance_ids = []
        logger.debug(f"In get_matches_against_current_frame, categories: {task.goal_semantic_id}")
        # first collect crops of instances found in the current frame
        for temp_id, inst_view in self.instance_memory.unprocessed_views.items():
            instance_id = self.instance_memory.temp_id_to_global_id.get(temp_id, -1)
            logger.debug(
                f"Processing instance {instance_id} with category {inst_view.category_id} (tmp id: {temp_id})."
            )
            if instance_id == -1:
                # logger.debug(f"No map cell assigned yet for this instance yet skipping.")
                # How can this happen? Objects get global id only if they are projected into the map. We might see an object, but it might not have any projection because:
                # 1. It is too far
                # 2. It is small that no point from it falls into any cell with enough confidence.
                continue
            views = self._get_valid_views([inst_view], task.goal_semantic_id, use_full_image)
            if len(views) > 0:
                logger.debug(
                    f">>>>>> Added to detections."
                )
                detections.append(views[0])
                instance_ids.append(instance_id)

        if len(detections) > 0:
            if task.type == "objectnav":
                confidences = self.match_to_category(
                    instance_ids, global_pose
                )
            else:
                confidences = self.match_images_to_goal(
                    detections,
                    task,
                )
            return np.array(confidences).reshape(-1, 1), instance_ids
        return [], []

    def match_to_category(
        self, instance_ids, global_pose
    ):
        #! myTODO: This is last steps global_pose, but I think it doesn't matter much. Ideally, I think we should do all these steps after SemMapModule.
        all_confidences = []
        for instance_id in instance_ids:
            assert instance_id != -1

            instance_views = self.instance_memory.instances[
                instance_id
            ].instance_views
            # pick a view with maximum object coverage
            best_view = np.argmax([view.object_coverage for view in instance_views])

            #1 Score based on coverage:
            # score = instance_views[best_view].object_coverage

            #2 Score based on distance:
            instance_pose = instance_views[best_view].pose
            global_xy = global_pose[:2].cpu()
            instance_xy = instance_pose[:2]
            score = 1 / (torch.norm(global_xy - instance_xy).item()+1)

            #3 Score based on distance and coverage:
            #! myTODO: Very important because we should not go to poses were only a couple of pixels are from the object.
            #! However, many times when we get close, we get better views which also have lower distances. 

            all_confidences.append(score)
        return all_confidences

    def match_images_to_goal(
        self,
        all_views,
        task,
    ):
        if task.type == "imagenav":
            all_confidences = self.match_image_to_image(
                all_views,
                task.goal_image_processed,
                task.goal_image_keypoints,
            )
        elif task.type == "languagenav":
            all_confidences = self.match_language_to_image(
                all_views,
                task.goal_description,
            )
        else:
            raise ValueError("Shouldn't happen")
        return all_confidences
    

    def get_matches_against_memory(
        self,
        task,
        use_full_image=False,
        global_pose=None
    ):
        """
        Compute matching scores from an image or language goal with each instance
        in the instance memory.
        """
        all_views = []
        instance_view_counts = []
        instance_ids = []
        for global_id, inst in self.instance_memory.instances.items():
            if inst.category_id != task.goal_semantic_id:
                continue
            views = self._get_valid_views(inst.instance_views, task.goal_semantic_id, use_full_image)
            if len(views) > 0:
                all_views.extend(views)
                instance_view_counts.append(len(views))
                instance_ids.append(global_id)

        if len(all_views) > 0:
            if task.type == "objectnav":
                all_confidences = self.match_to_category(
                    instance_ids, global_pose
                )
                all_confidences = np.array(all_confidences).reshape(-1, 1)
            else:
                all_confidences = self.match_images_to_goal(
                    all_views,
                    task,
                )
                # unflatten based on number of views per instance
                # all_confidences = np.concatenate(all_confidences, 0)
                all_confidences = np.split(
                    all_confidences, np.cumsum(instance_view_counts)[:-1]
                )
            return all_confidences, instance_ids
        return [], []

    @torch.no_grad()
    def match_image_to_image(
        self,
        rgb_images: List[np.ndarray],
        goal_image: torch.Tensor,
        goal_image_keypoints: Dict[str, Any],
    ):
        """Computes and describes keypoints using SuperPoint and matches
        keypoints between an RGB image and a goal image using SuperGlue.
        Either goal_image or goal_image_keypoints must be provided.
        Returns:
            tensor of goal image keypoints
            tensor of rgb image keypoints
            tensor of keypoint matches
            tensor of match confidences
        """
        assert isinstance(goal_image, torch.Tensor) # Already prreprocessed
        assert isinstance(rgb_images, list)
        assert isinstance(rgb_images[0], np.ndarray)

        all_confidences = []

        # TODO Can we batch this for loop to speed it up? It is a bottleneck
        logger.debug("Computing matching score with each view...")
        for i in range(len(rgb_images)):
            rgb_image_processed = self._preprocess_image(rgb_images[i])

            matcher_inputs = {
                "image0": goal_image,
                "image1": rgb_image_processed,
                **goal_image_keypoints,
            }
            pred = self.matcher(matcher_inputs)

            matches = pred["matches0"].cpu().numpy()
            confidence = pred["matching_scores0"].cpu().numpy()
            self._visualize(matcher_inputs, pred, f"{self.step}_{i}")

            confidence = confidence[matches != -1].sum().item()
            all_confidences.append(confidence)
        return all_confidences

    @torch.no_grad()
    def match_language_to_image(self, views_orig, language_goal, **kwargs):
        """Compute matching scores from a language goal to images."""
        batch_size = 64
        language_goal = language_goal.replace("Instruction: ", "")
        language_goal = clip.tokenize(language_goal).to(self.device)
        language_goal = self.clip_model.encode_text(language_goal)
        # get clip embedding for views with a batch size of batch_size

        views = views_orig
        if views[0].shape[0] == 3:
            views = torch.stack(
                [
                    self.clip_preprocess(
                        ToPILImage()(v.transpose(2, 1, 0).astype(np.uint8))
                    )
                    for v in views
                ],
                dim=0,
            )
        else:
            views = torch.stack(
                [self.clip_preprocess(ToPILImage()(v.astype(np.uint8))) for v in views],
                dim=0,
            )
        view_embeddings = torch.cat(
            [
                self.clip_model.encode_image(v.to(self.device))
                for v in views.split(batch_size)
            ],
            dim=0,
        )
        # normalize the embeddings
        view_embeddings = view_embeddings / view_embeddings.norm(dim=-1, keepdim=True)
        language_goal = language_goal / language_goal.norm(dim=-1, keepdim=True)
        # compute cosines similarity
        similarity = (language_goal @ view_embeddings.T).squeeze(0)
        return similarity.detach().cpu().numpy().flatten()

    def get_best_match(self, scores, instance_ids, score_thresh):
        """instance_ids are global"""
        sorted_inst_ids = np.argsort(scores)[::-1]
        idx = 0
        logger.debug(
            f"Getting best match. Scores: {scores}, instance_ids: {instance_ids}, score threshold: {score_thresh}."
        )
        while (
            idx < len(sorted_inst_ids) and scores[sorted_inst_ids[idx]] >= score_thresh
        ):
            inst_idx = sorted_inst_ids[idx]
            idx += 1
            logger.debug(
                f"Trying to localize instance {instance_ids[inst_idx]} with score {scores[inst_idx]}"
            )
            best_instance_id = instance_ids[inst_idx]
            if best_instance_id == -1:
                # * The reason that this might happen is sometimes someobjects are overriding that instance when projecting into 2D (object with highest point is chosen in a specifc cell).
                logger.debug("No id provided for this instance. Skipping.")
                continue
            # * We does not check it goal map is non-empty here. Somewehre else, I should make sure we can navigate to this goal.
            return True, best_instance_id

        logger.debug("Goal does not match any instance.")
        return False, None

    def aggregate_scores_per_instance(self, confidences, agg_fn):
        agg_scores = []
        for inst_idx, inst_confidences in enumerate(confidences):
            if agg_fn == "max":
                agg_scores.append(max(inst_confidences))
            elif agg_fn == "mean":
                agg_scores.append(np.mean(inst_confidences))
            elif agg_fn == "median":
                agg_scores.append(np.median(inst_confidences))
            else:
                raise NotImplementedError
            # logger.debug(f"Instance {inst_idx+1} score: {max(inst_view_scores)}")
            logger.debug(f"Instance {inst_idx+1} score: {agg_scores[-1]}")
        return agg_scores

    def search_for_goal(
        self,
        task,
        match_memory,
        global_pose,
        agg_fn: str = "max",
        score_thresh=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Optional[int]]:
        if score_thresh is None:
            score_thresh = self.score_thresh[task.type]

        mem_match_confidences, mem_match_instance_ids = [], []
        if match_memory:
            self.log.info("--------Matching against memory!--------")
            (
                mem_match_confidences,
                mem_match_instance_ids,
            ) = self.get_matches_against_memory(
                task,
                use_full_image=True,
                global_pose=global_pose,
            )

        obs_match_confidences, obs_match_instance_ids = (
            self.get_matches_against_current_frame(
                task,
                use_full_image=False,
                global_pose=global_pose,
            )
        )

        #! myTODO: Should I overwrite Mem with obs, or otherwise?
        inst_goal_found = False
        inst_goal_id = None

        if len(mem_match_confidences) > 0:
            logger.debug(
                f"Matching with memory: {len(mem_match_confidences)} instances"
            )
            agg_scores = self.aggregate_scores_per_instance(
                mem_match_confidences, agg_fn
            )
            if len(agg_scores) > 0:
                inst_goal_found, inst_goal_id = self.get_best_match(
                    agg_scores, mem_match_instance_ids, score_thresh
                )
            if inst_goal_found is True:
                logger.info(f"Goal instance {inst_goal_id} found by matching with memory.")
            else:
                logger.debug(f"No matches found in the memory")

        if inst_goal_found is False and len(obs_match_confidences) > 0:
            logger.debug(
                f"Matching with observation: {len(obs_match_confidences)} instances"
            )
            logger.debug(
                f"Global instance ids: {obs_match_instance_ids}"
            )
            agg_scores = self.aggregate_scores_per_instance(
                obs_match_confidences, agg_fn
            )

            inst_goal_found, inst_goal_id = self.get_best_match(
                agg_scores, obs_match_instance_ids, score_thresh
            )
            if inst_goal_found is True:
                logger.debug(
                    f"Goal instance {inst_goal_id} found in this step by matching with observation."
                )
            else:
                logger.debug(f"No matches found with observation")

        return inst_goal_found, inst_goal_id


    def _get_valid_views(self, inst_views, category, use_full_image):
        views = []
        for inst_view in inst_views:
            if inst_view.category_id != category:
                continue
            # Note: Using bbox shape instead of cropped image shape, because cropped image doesn't always add a fixed padding.
            bbox_shape =  inst_view.bbox[1] - inst_view.bbox[0]
            logger.debug(f"Total pixels in cropped image: {bbox_shape.prod()} ? {MIN_PIXELS}")
            logger.debug(f"Minimum edge size in cropped image: {bbox_shape} ? {MIN_EDGE} : {(bbox_shape < MIN_EDGE).any()}")
            if bbox_shape.prod() < MIN_PIXELS or (bbox_shape < MIN_EDGE).any():
                continue
            if use_full_image:
                img = self.instance_memory.images[inst_view.timestep].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
            else:
                img = inst_view.cropped_image

            views.append(img)
        return views