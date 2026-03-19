"""
script to calculate mean average precision for 3d radar point clouds clusters.
copied from torchmetrics.
"""

from typing import Any, List, Dict, Literal, Tuple, Optional
import torch
from torch import Tensor
from torchmetrics import Metric
import contextlib
from lightning_utilities import apply_to_collection
from roadside_radar_seg.evaluation import RadarCOCO as coco
from roadside_radar_seg.evaluation import RadarCOCOeval as cocoeval
import numpy as np
import io
from roadside_radar_seg.configs import configurable
from torchmetrics import ConfusionMatrix


class RadarMeanAveragePrecision(Metric):

    @configurable
    def __init__(
        self,
        *,
        iou_thresholds=None,
        rec_thresholds=None,
        class_metrics=True,
        extended_summary=False,
        average="macro",
        max_detection_thresholds=[100],
        **kwargs: Any,
    ) -> None:

        super().__init__(**kwargs)
        if iou_thresholds is not None and not isinstance(iou_thresholds, list):
            raise ValueError(
                f"Expected argument `iou_thresholds` to either be `None` or a list of floats but got {iou_thresholds}"
            )
        self.iou_thresholds = iou_thresholds
        self.average = average
        self.class_metrics = class_metrics
        self.extended_summary = extended_summary
        self.max_detection_thresholds = max_detection_thresholds  # [1, 10, 100]

        if rec_thresholds is not None and not isinstance(rec_thresholds, list):
            raise ValueError(
                f"Expected argument `rec_thresholds` to either be `None` or a list of floats but got {rec_thresholds}"
            )
        self.rec_thresholds = (
            rec_thresholds or torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()
        )

        self.add_state("detection_clusters", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_clusters", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)

    @classmethod
    def from_config(cls, cfg):

        return {
            "iou_thresholds": cfg.EVALUATION.IOU_THRESHOLDS,
            "rec_thresholds": cfg.EVALUATION.REC_THRESHOLDS,
            "class_metrics": cfg.EVALUATION.CLASS_METRICS,
            "extended_summary": cfg.EVALUATION.EXTENDED_SUMMARY,
            "average": cfg.EVALUATION.AVERAGE,
            "max_detection_thresholds": cfg.EVALUATION.MAX_DETECTION_THRESHOLD,
        }

    def update(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ) -> None:

        for item in preds:
            pred_labels = item["labels"]  # tensor
            pred_scores = item["scores"]  # tensor
            pred_clusters = item["clusters"]  # tensor

            assert len(pred_labels) == len(pred_scores) == len(pred_clusters)

            assert isinstance(pred_labels, torch.Tensor)
            assert isinstance(pred_scores, torch.Tensor)
            assert isinstance(pred_clusters, list)

            self.detection_clusters.append(pred_clusters)
            self.detection_scores.append(pred_scores)
            self.detection_labels.append(pred_labels)

        for item in target:
            gt_labels = item["labels"]
            gt_clusters = item["clusters"]

            assert isinstance(gt_labels, torch.Tensor)
            assert isinstance(gt_clusters, list)

            self.groundtruth_clusters.append(gt_clusters)
            self.groundtruth_labels.append(gt_labels)

    # @property
    # def coco(self) -> object:
    #     """Returns the coco module for the given backend, done in this way to make metric picklable."""
    #     coco, _, _ = _load_backend_tools(self.backend)
    #     return coco

    def compute(self) -> dict:

        coco_preds, coco_target = self._get_coco_datasets(average=self.average)

        result_dict = {}

        with contextlib.redirect_stdout(io.StringIO()):
            prefix = ""
            coco_eval = cocoeval(coco_target, coco_preds, iouType="pointcloud")
            # coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
            # coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
            coco_eval.params.maxDets = self.max_detection_thresholds

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
            result_dict.update(self._coco_stats_to_tensor_dict(stats, ""))

            summary = {}
            if self.extended_summary:
                summary = {
                    f"{prefix}ious": apply_to_collection(
                        coco_eval.ious,
                        np.ndarray,
                        lambda x: torch.tensor(x, dtype=torch.float32),
                    ),
                    f"{prefix}precision": torch.tensor(coco_eval.eval["precision"]),
                    f"{prefix}recall": torch.tensor(coco_eval.eval["recall"]),
                    f"{prefix}scores": torch.tensor(coco_eval.eval["scores"]),
                }
            result_dict.update(summary)

            if self.class_metrics:
                if self.average == "micro":
                    # since micro averaging have all the data in one class, we need to reinitialize the coco_eval
                    # object in macro mode to get the per class stats
                    coco_preds, coco_target = self._get_coco_datasets(average="macro")
                    coco_eval = self.cocoeval(
                        coco_target, coco_preds, iouType="pointcloud"
                    )
                    coco_eval.params.iouThrs = np.array(
                        self.iou_thresholds, dtype=np.float64
                    )
                    coco_eval.params.recThrs = np.array(
                        self.rec_thresholds, dtype=np.float64
                    )
                    coco_eval.params.maxDets = self.max_detection_thresholds

                map_per_class_list = []
                mar_100_per_class_list = []
                for class_id in self._get_classes():
                    coco_eval.params.catIds = [class_id]
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        coco_eval.summarize()
                        class_stats = coco_eval.stats

                    map_per_class_list.append(torch.tensor([class_stats[0]]))
                    mar_100_per_class_list.append(torch.tensor([class_stats[8]]))

                map_per_class_values = torch.tensor(
                    map_per_class_list, dtype=torch.float32
                )
                mar_100_per_class_values = torch.tensor(
                    mar_100_per_class_list, dtype=torch.float32
                )
            else:
                map_per_class_values = torch.tensor([-1], dtype=torch.float32)
                mar_100_per_class_values = torch.tensor([-1], dtype=torch.float32)
            prefix = ""
            result_dict.update(
                {
                    f"{prefix}map_per_class": map_per_class_values,
                    f"{prefix}mar_100_per_class": mar_100_per_class_values,
                },
            )
        result_dict.update(
            {"classes": torch.tensor(self._get_classes(), dtype=torch.int32)}
        )

        return result_dict

    @staticmethod
    def _coco_stats_to_tensor_dict(
        stats: List[float], prefix: str
    ) -> Dict[str, Tensor]:
        """Converts the output of COCOeval.stats to a dict of tensors."""
        return {
            f"{prefix}map": torch.tensor([stats[0]], dtype=torch.float32),
            f"{prefix}map_50": torch.tensor([stats[1]], dtype=torch.float32),
            f"{prefix}map_75": torch.tensor([stats[2]], dtype=torch.float32),
            f"{prefix}map_small": torch.tensor([stats[3]], dtype=torch.float32),
            f"{prefix}map_medium": torch.tensor([stats[4]], dtype=torch.float32),
            f"{prefix}map_large": torch.tensor([stats[5]], dtype=torch.float32),
            f"{prefix}mar_1": torch.tensor([stats[6]], dtype=torch.float32),
            f"{prefix}mar_10": torch.tensor([stats[7]], dtype=torch.float32),
            f"{prefix}mar_100": torch.tensor([stats[8]], dtype=torch.float32),
            f"{prefix}mar_small": torch.tensor([stats[9]], dtype=torch.float32),
            f"{prefix}mar_medium": torch.tensor([stats[10]], dtype=torch.float32),
            f"{prefix}mar_large": torch.tensor([stats[11]], dtype=torch.float32),
        }

    def _get_coco_datasets(
        self, average: Literal["macro", "micro"]
    ) -> Tuple[object, object]:
        """Returns the coco datasets for the target and the predictions."""
        if average == "micro":
            # for micro averaging we set everything to be the same class
            groundtruth_labels = apply_to_collection(
                self.groundtruth_labels, Tensor, lambda x: torch.zeros_like(x)
            )
            detection_labels = apply_to_collection(
                self.detection_labels, Tensor, lambda x: torch.zeros_like(x)
            )
        else:
            groundtruth_labels = self.groundtruth_labels
            detection_labels = self.detection_labels

        coco_target, coco_preds = coco(), coco()

        coco_target.dataset = self._get_coco_format(
            labels=groundtruth_labels,
            clusters=(
                self.groundtruth_clusters
                if len(self.groundtruth_clusters) > 0
                else None
            ),
        )
        coco_preds.dataset = self._get_coco_format(
            labels=detection_labels,
            clusters=(
                self.detection_clusters if len(self.detection_clusters) > 0 else None
            ),
            scores=self.detection_scores,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

        return coco_preds, coco_target

    def _get_coco_format(
        self,
        labels: List[torch.Tensor],
        clusters: Optional[List[List[torch.Tensor]]] = None,
        scores: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.
        bounding box is replaced with point clusters.

        Format is defined at
        https://cocodataset.org/#format-data

        """
        pcds = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        for pcd_id, pcd_labels in enumerate(labels):
            if clusters is not None:
                pcd_clusters = clusters[pcd_id]  # list of tensors.

            pcd_labels = pcd_labels.cpu().tolist()

            pcds.append({"id": pcd_id})

            for k, pcd_label in enumerate(pcd_labels):
                if clusters is not None:
                    pcd_cluster = pcd_clusters[k]

                pcd_label = int(pcd_label)

                annotation = {
                    "id": annotation_id,
                    "pcd_id": pcd_id,
                    "area": 0,
                    "category_id": pcd_label,
                    "iscrowd": 0,
                }

                if clusters is not None:
                    annotation["clusters"] = pcd_cluster

                if scores is not None:
                    score = scores[pcd_id][k].cpu().tolist()
                    if not isinstance(score, float):
                        raise ValueError(
                            f"Invalid input score of sample {pcd_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {"pcds": pcds, "annotations": annotations, "categories": classes}

    def _get_classes(self) -> List:
        """Return a list of unique classes found in ground truth and detection data."""
        if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
            return (
                torch.cat(self.detection_labels + self.groundtruth_labels)
                .unique()
                .cpu()
                .tolist()
            )
        return []


def build_map_evaluator(cfg):
    device = cfg.MODEL.DEVICE
    return RadarMeanAveragePrecision(cfg).to(device)


def build_cm_evaluator(cfg):
    device = cfg.MODEL.DEVICE
    return ConfusionMatrix(
        task="multiclass", num_classes=cfg.MODEL.SEGM_HEAD.NUM_CLASSES
    ).to(device)
