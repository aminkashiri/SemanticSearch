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
        score_func: str,
        num_sem_categories: int,
        config: Dict[str, Any],
        default_vis_dir: str,
        print_images: bool,
        instance_memory: InstanceMemory,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

        assert score_func in ["confidence_sum", "match_count"]
        self.score_func = score_func
        self.num_sem_categories = num_sem_categories

        # generate clip embeddings by loading clip model
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device)
        self.goto_past_pose = config.goto_past_pose
        self.instance_memory = instance_memory

    def get_matches_against_current_frame(
        self,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        use_full_image=False,
        categories=None,
        **kwargs,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        detected in the current frame.
        """
        instance_memory = self.instance_memory
        # TODO We should restrict detections in the current frame by category
        detections = []
        instance_ids = []
        # first collect crops of instances found in the current frame
        for local_instance_id, inst_view in instance_memory.unprocessed_views.items():
            if categories is not None and inst_view.category_id not in categories:
                continue
            if (
                inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1]
                < MIN_PIXELS
                or (np.array(inst_view.cropped_image.shape[0:2]) < MIN_EDGE).any()
            ):
                continue
            if use_full_image:
                img = instance_memory.images[-1].cpu().numpy()
            else:
                img = inst_view.cropped_image
            detections.append(img)
            instance_ids.append(local_instance_id)

        confidences = []
        if len(detections) > 0:
            confidences = self.match_images_to_goal(
                detections,
                matching_fn,
                step,
                use_full_image=use_full_image,
                image_goal=image_goal,
                language_goal=language_goal,
                **kwargs,
            )
        try:
            return np.array(confidences).reshape(-1,1), np.array(instance_ids)
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()

    def match_images_to_goal(
        self,
        all_views,
        matching_fn,
        step,
        use_full_image=False,
        image_goal=None,
        language_goal=None,
        **kwargs,
    ):
        all_confidences = []
        if image_goal is not None:
            all_confidences = matching_fn(
                all_views,
                goal_image=image_goal,
                goal_image_keypoints=kwargs["goal_image_keypoints"],
                use_full_image=use_full_image,
                step=1000 * step,
            )
        elif language_goal is not None:
            all_confidences = matching_fn(
                all_views,
                language_goal,
            )
        else:
            all_confidences = [1] * len(all_views)
        return all_confidences

    def get_matches_against_memory(
        self,
        matching_fn,
        step,
        image_goal=None,
        language_goal=None,
        use_full_image=False,
        categories=None,
        **kwargs,
    ):
        """
        Compute matching scores from an image or language goal with each instance
        in the instance memory.
        """
        instance_memory = self.instance_memory
        all_confidences = []
        instances = instance_memory.instance_views
        all_views = []
        instance_view_counts = []
        steps_per_view = []
        instance_ids = []
        for (inst_key, inst) in instances.items():
            if categories is not None and inst.category_id not in categories:
                continue
            inst_views = inst.instance_views
            views_added = 0
            for view_idx, inst_view in enumerate(inst_views):
                if (
                    inst_view.cropped_image.shape[0] * inst_view.cropped_image.shape[1]
                    < MIN_PIXELS
                    or (np.array(inst_view.cropped_image.shape[0:2]) < MIN_EDGE).any()
                ):
                    continue
                if use_full_image:
                    img = instance_memory.images[inst_view.timestep].cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))
                else:
                    img = inst_view.cropped_image

                all_views.append(img)
                views_added += 1
                steps_per_view.append(1000 * step + 10 * inst_key + view_idx)
            if views_added > 0:
                instance_view_counts.append(views_added)
                instance_ids.append(inst_key)

        if len(all_views) > 0:
            all_confidences = self.match_images_to_goal(
                all_views,
                matching_fn,
                step,
                use_full_image=use_full_image,
                image_goal=image_goal,
                language_goal=language_goal,
                **kwargs,
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
        rgb_image: Union[np.ndarray, List[np.ndarray]],
        goal_image: Union[np.ndarray, torch.Tensor],
        rgb_image_keypoints: Optional[Dict[str, Any]] = None,
        goal_image_keypoints: Optional[Dict[str, Any]] = None,
        use_full_image: bool = False,
        step: Optional[int] = None,
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
        if isinstance(rgb_image, np.ndarray) and len(rgb_image.shape) == 3:
            rgb_image_batched = [rgb_image]
        else:
            rgb_image_batched = rgb_image
            assert rgb_image_keypoints is None

        all_confidences = []

        # TODO Can we batch this for loop to speed it up? It is a bottleneck
        logger.debug("Computing matching score with each view...")
        for i in range(len(rgb_image_batched)):
        # for i in tqdm(range(len(rgb_image_batched))):
            if goal_image_keypoints is None:
                goal_image_keypoints = {}
            if rgb_image_keypoints is None:
                rgb_image_keypoints = {}

            if isinstance(goal_image, np.ndarray):
                goal_image_processed = self._preprocess_image(goal_image)
            else:
                goal_image_processed = goal_image
            if isinstance(rgb_image_batched[i], np.ndarray):
                if rgb_image_batched[i].shape[0] == 3:
                    rgb_image_batched[i] = rgb_image_batched[i].transpose(1,2,0)
                rgb_image_processed = self._preprocess_image(
                    rgb_image_batched[i].astype(np.uint8)
                )
            else:
                rgb_image_processed = rgb_image_batched[i]

            matcher_inputs = {
                "image0": goal_image_processed,
                "image1": rgb_image_processed,
                **goal_image_keypoints,
                **rgb_image_keypoints,
            }
            pred = self.matcher(matcher_inputs)

            matches = pred["matches0"].cpu().numpy()
            confidence = pred["matching_scores0"].cpu().numpy()
            self._visualize(matcher_inputs, pred, step + i)

            if "keypoints0" in matcher_inputs:
                goal_keypoints = matcher_inputs["keypoints0"]
            else:
                goal_keypoints = pred["keypoints0"]

            if "keypoints1" in matcher_inputs:
                rgb_keypoints = matcher_inputs["keypoints1"].cpu().numpy()
            else:
                rgb_keypoints = [pred["keypoints1"][0].cpu().numpy()]
            if isinstance(rgb_image, np.ndarray) and len(rgb_image.shape) == 3:
                return goal_keypoints, rgb_keypoints, matches, confidence

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
                [self.clip_preprocess(ToPILImage()(v.transpose(2,1,0).astype(np.uint8))) for v in views],
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

    def get_best_match(self, scores, instance_ids, instance_map, score_thresh):
        instance_goal_found = False
        goal_inst = None
        sorted_inst_ids = np.argsort(scores)[::-1]
        idx = 0
        logger.debug(f"Getting best match. Scores: {scores}, instance_ids: {instance_ids}, score threshold: {score_thresh}.")
        while (
            idx < len(sorted_inst_ids) and scores[sorted_inst_ids[idx]] > score_thresh
        ):
            inst_idx = sorted_inst_ids[idx]
            idx += 1
            logger.debug(
                f"Trying to localize instance {inst_idx + 1} with score {scores[inst_idx]}"
            )
            if instance_ids is None:
                best_instance_id = inst_idx + 1
            else:
                best_instance_id = instance_ids[inst_idx]
            if instance_ids[inst_idx] == -1:
                #* The reason that this might happen is sometimes someobjects are overriding that instance when projecting into 2D (object with highest point is chosen in a specifc cell).
                logger.debug("Found the goal in current observation, but is not in instance map. Skipping.")
                continue
            inst_map_idx = instance_map == best_instance_id
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))

            if not self.goto_past_pose:
                goal_map_temp = (instance_map[inst_map_idx] == best_instance_id).float()
                if goal_map_temp.any():
                    instance_goal_found = True
                    goal_inst = best_instance_id
                    logger.debug(f"Instance {goal_inst} will be the goal")
                    return instance_goal_found, goal_inst
                else:
                    logger.debug("Instance was seen, but not present in local map.")
            else:
                #! TODODODODODOOD myTODO . FILL TODAY. THIS IS NOT OK. EVEN IF I CHECK IT IS IN THE LOCAL MAP, IT MIGHT NOT BE WHEN CHECKING FOR POSE
                # we are ok with object not being on map when using agent pose as target
                return True, best_instance_id

        if idx == len(sorted_inst_ids):
            logger.debug("Goal image does not match any instance.")

        return instance_goal_found, goal_inst

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

    def get_goal_map_from_goal_instance(
        self, instance_map, lmb, goal_inst
    ):
        #! myTODO: We are checking goal_inst and instance_goal_found here before calling the function, so I should be able to remove instance_goal_found as input.
        if goal_inst is None:
            logger.error(f"Goal instance {goal_inst} not found")
            raise Exception(f"Goal instance {goal_inst} not found")
        global_view_loc = None

        if self.goto_past_pose:
            instance_memory = self.instance_memory
            instance_views = instance_memory.instance_views[goal_inst].instance_views
            # pick a view with maximum object coverage
            best_view = np.argmax([view.object_coverage for view in instance_views])
            pose = instance_views[best_view].pose
            curr_x, curr_y, curr_o, gy1, _, gx1, _ = pose.tolist()


            # Past pose closest to the goal instance, new code
            inst_map_idx = instance_map == goal_inst
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
            goal_map = (instance_map[inst_map_idx] == goal_inst).to(torch.float)
            
            #! The output goal pose is the actual index in global map
            global_view_loc = [int(curr_y * 100.0 / 5) , int(curr_x * 100.0 / 5), curr_o]

            # from home_robot.utils.visualization import visualize_map
            # import os
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            # visualize_map(goal_map.shape, current_dir, f"goal_instance_{goal_inst}_coverage_map.png" , goal_map=goal_map.cpu().numpy())

            logger.debug(f">>> Goal instance {goal_inst} best view loc is: {global_view_loc}, with coverage {instance_views[best_view].object_coverage}. Returning goal_pose in addition to goal_map.")


        else:
            inst_map_idx = instance_map == goal_inst
            inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
            goal_map = (instance_map[inst_map_idx] == goal_inst).to(torch.float)
            logger.info(f">>> Returning goal_map for instance: {goal_inst}.")

        goal_map = goal_map.cpu().numpy()

        return goal_map, global_view_loc


    def select_and_localize_instance(
        self,
        instance_map: torch.Tensor,
        lmb: torch.Tensor,  # local map boundaries
        confidence: torch.Tensor,
        frame_matches_local_instance_ids: List,
        local_id_to_global_id_map: Optional[Dict],
        all_confidences: List = None,
        instance_ids: List = None,
        score_thresh: float = 0.0,
        agg_fn: str = "max",
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Optional[int]]:
        """Select and localize an instance given computed matching scores."""
        goal_map = None
        goal_pose = None
        instance_goal_found = False
        goal_inst = None

        if all_confidences is not None and len(all_confidences) > 0:
            logger.debug(f"Matching with memory: {len(all_confidences)} instances")
            agg_scores = self.aggregate_scores_per_instance(
                all_confidences, agg_fn
            )
            if len(agg_scores) > 0:
                instance_goal_found, goal_inst = self.get_best_match(
                    agg_scores, instance_ids, instance_map, score_thresh
                )
        if instance_goal_found is True:
            logger.info(f"Goal instance {goal_inst} found by matching with memory.")
        else:
            logger.debug(f"No matches found in the memory")
        

        if instance_goal_found is False and confidence is not None and len(confidence) > 0:
            logger.debug(f"Matching with observation: {len(confidence)} instances")
            global_instance_ids = [
                local_id_to_global_id_map.get(i, -1)
                for i in frame_matches_local_instance_ids
            ]
            logger.debug(f"Global instance ids: {global_instance_ids}, local instance ids: {frame_matches_local_instance_ids}")
            agg_scores = self.aggregate_scores_per_instance(
                confidence, agg_fn
            )
            instance_goal_found, goal_inst = self.get_best_match(
                agg_scores, global_instance_ids, instance_map, score_thresh
            )
            if instance_goal_found is True:
                logger.debug(f"Goal instance {goal_inst} found in this step by matching with observation.")
            else:
                logger.debug(f"No matches found with observation")


        if goal_inst is not None and instance_goal_found is True:
            logger.info(f"Localizing found goal instance")
            goal_map, goal_pose = self.get_goal_map_from_goal_instance(
                instance_map, lmb, goal_inst
            )
        else:
            logger.debug(f"Didn't find any match in this step")
        
        return goal_map, goal_pose, instance_goal_found, goal_inst
