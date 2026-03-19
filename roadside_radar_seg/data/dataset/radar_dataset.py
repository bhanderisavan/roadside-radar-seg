#! /usr/bin/env python3

from typing import Tuple, List, Type
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pathlib
import json
import torch
import numpy.lib.recfunctions as rfn
import ast
from roadside_radar_seg.configs import configurable
from ..radar_background_subtraction import build_radar_bg_subtractor
from roadside_radar_seg.utils import PcdHelper
from . import builtin_meta as metadata
from copy import deepcopy

class RadarDataset(Dataset):
    """
    a dataset class for radar dataset to feed to torch dataloader
    """

    @configurable
    def __init__(
        self,
        *,
        root: str,
        field_names: List[str],
        device: str,
        bg_subtractor: Type,
        bg_sub_grid_folder_path: str,
        max_abs_v_thresh: float,
        max_depth: float,
        dynamic_v_thresh: float,  # points with abs(v) < dynamic_v_thresh will be considered as static points.
    ):
        """
        Args:
            root (str): root to the dataset (train or val)
            field_names (list) : list of radar point fields to keep in pcd. defaults to None.
                if None, all the fields are returned.
            transform (callable): transformations to be performed on the data.
            device (str) : cpu or cuda.
        """

        self.root = Path(root) if not isinstance(root, pathlib.PosixPath) else root

        self.field_names = field_names

        assert self.root.is_dir(), f"root must be a path to the dataset dir"

        self.bg_subtractor = bg_subtractor

        # path to annotations
        self.targets_dir = self.root.joinpath("annotations")
        # path to radar pcds
        self.pcds_dir = self.root.joinpath("pcds")

        # list of full paths of annotations
        self.targets = sorted(self.targets_dir.glob("*.json"))
        # list of full paths of radar pcds
        self.pcds = sorted(self.pcds_dir.glob("*.pcd"))

        self.pcd_helper = PcdHelper()
        self.device = device

        if self.bg_subtractor is not None:
            assert bg_sub_grid_folder_path is not None

        self.bg_sub_grid_folder_path = bg_sub_grid_folder_path

        if "index" not in self.field_names:
            raise ValueError(
                "input field names does not contain 'index' field. Please add it."
            )

        assert len(self.targets) == len(
            self.pcds
        ), "Number of pcds and targets must be same"

        self.max_abs_v_thresh = max_abs_v_thresh
        self.max_depth = max_depth
        self.dynamic_v_thresh = dynamic_v_thresh

    @classmethod
    def from_config(cls, cfg, split):

        assert split in ["train", "val"], f"split cannot be {split}"

        bg_subtractor = None

        if split == "train" and cfg.TRAIN.USE_BGSUB:
            bg_subtractor = build_radar_bg_subtractor(cfg)
        if split == "val" and cfg.VAL.USE_BGSUB:
            bg_subtractor = build_radar_bg_subtractor(cfg)

        bg_sub_grid_folder_path = cfg.BGSUB.GRID_FOLDER_PATH

        root = cfg.TRAIN.DATASET_PATH
        if split == "val":
            root = cfg.VAL.DATASET_PATH

        return {
            "root": root,
            "field_names": cfg.INPUT.INPUT_FIELDS,
            "device": cfg.MODEL.DEVICE,
            "bg_subtractor": bg_subtractor,
            "bg_sub_grid_folder_path": bg_sub_grid_folder_path,
            "max_abs_v_thresh": cfg.INPUT.MAX_ABS_VELOCITY,
            "max_depth": cfg.INPUT.MAX_DEPTH,
            "dynamic_v_thresh": cfg.INPUT.DYNAMIC_V_THRESHOLD,
        }

    def _get_timestamp_from_name(self, name: str) -> str:
        """returns timestamp part of the filename

        Args:
            name: filename with extension

        returns:
            str: timestamp part of the name

        """

        name = name.split("__")[-1]
        stamp = name.split(".")[0]

        return stamp

    def _get_bg_grid_idx_from_filename(self, filename: str) -> int:

        stamp = self._get_timestamp_from_name(filename)
        bg_idx = stamp.split("_")[-1][2:]

        return int(bg_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, np.recarray]:
        """
        returns:
            pcd_cloud : radar pcd pointcloud with dynamic points and static foreground points.
            annotation_data : target dict of ground truth information
        """
        name = None  # for debugging, when we want to get a sample using its name and not its index.
        if isinstance(idx, tuple):
            idx, name = idx

        if name is not None:
            pcd_name = f"radar_01__{name.split('.')[0].split('__')[-1]}.pcd"
            pcd_path = list(self.pcds_dir.glob(pcd_name))[0]
        else:
            pcd_path = self.pcds[idx]

        # sanity check timestamp matching
        pcd_timestamp = self._get_timestamp_from_name(pcd_path.name)

        # raw radar cloud.
        pcd_cloud = self.pcd_helper.read_radar_pcd(str(pcd_path))

        #all_points = deepcopy(pcd_cloud)

        # clip the cloud to camera fov because we have annotation only within camera fov
        pcd_cloud = pcd_cloud[pcd_cloud["u"] >= 0]
        pcd_cloud = pcd_cloud[pcd_cloud["u"] <= 1920]
        pcd_cloud = pcd_cloud[pcd_cloud["v"] >= 0]
        pcd_cloud = pcd_cloud[pcd_cloud["v"] <= 1216]

        # threshold the cloud
        # max velocity threshold
        pcd_cloud = pcd_cloud[abs(pcd_cloud["range_rate"]) <= self.max_abs_v_thresh]
        # max depth threshold
        pcd_cloud = pcd_cloud[pcd_cloud["x"] <= self.max_depth]

        # pcd_cloud = pcd_cloud[abs(pcd_cloud["y"]) <= 50]
        pcd_cloud = pcd_cloud[abs(pcd_cloud["z"]) <= 10]

        # dynamic - static separation
        pcd_cloud_dynamic = pcd_cloud[
            abs(pcd_cloud["range_rate"]) >= self.dynamic_v_thresh
        ]

        # pcd_cloud_static = pcd_cloud[
        #     abs(pcd_cloud["range_rate"]) < self.dynamic_v_thresh
        # ]

        # temp - keep only dynamic points if bg sub is not active.
        # empty array, will be  replaced with bgsub out put if bgsub is active, else only dynamic points will be considered.
        pcd_cloud_static = pcd_cloud[
            abs(pcd_cloud["range_rate"]) > self.max_abs_v_thresh
        ]

        if self.bg_subtractor is not None:
            # idx of bg subtraction grid for this frame. - extracted from the filename of the pcd.
            # 01 -> auwaldsee, 02 -> dk parking.
            bg_idx = self._get_bg_grid_idx_from_filename(pcd_path.name)

            # bg is applied in the data collection only id bg_idx > 0.
            if bg_idx > 0:

                # 1 -> "01"
                dst_bg_idx = str(bg_idx).zfill(2)

                # .npy file representing the bg sub grid.
                # self.bg_sub_grid_folder_path contains bgsub grids with naming convention of id__name.npy.
                grid_path = list(
                    Path(self.bg_sub_grid_folder_path).glob(f"{dst_bg_idx}__*")
                )[0]

                assert grid_path.is_file()

                # bg sub grid
                grid = np.load(str(grid_path))
                pcd_cloud_static = pcd_cloud[
                    abs(pcd_cloud["range_rate"]) < self.dynamic_v_thresh
                ]

                pcd_cloud_static, _ = self.bg_subtractor.perform_bg_sub(
                    grid, pcd_cloud_static
                )

        if name is not None:
            target_name = f"{name.split('.')[0]}.json"
            target_path = list(self.targets_dir.glob(target_name))[0]
        else:
            target_path = self.targets[idx]

        # preparing the target dict using annotation json.
        annotation_data = {}
        # annotation data is dict with following keys and vals :
        """
        {
            "name": "name of the radar pcd file.",
            "date_captured": datetime when the data was recorded,
            "pcd_id": "index of the pcd from the dataset.", # dummy
            "boxes": "torch.Tensor of the bounding boxes in image coordinate frame.",
            "labels": "torch.Tensor of category id labels for each objects in current pcd."
                    Note: These labels are remapped to 6 classes. so the range is [1-6], and the child and truck are removed.,
                    see class_names.INFRA3DRC_TO_CLASS_ID_REMAP for more details.
            "points": "list of np.recarray representing radar points for each object",
        }
        """

        # target are only provided for training and validation phase
        with open(str(target_path.absolute()), "r") as f:
            target = json.load(f)

        # sanity check for ensuring that the annotation and image time stamps match
        json_timestamp = self._get_timestamp_from_name(target_path.name)
        if not pcd_timestamp == json_timestamp:
            raise ValueError(
                f"pcd timestamp  {pcd_timestamp} and json timestamp {json_timestamp} does not match"
            )

        # parsing the json annotations
        anns = target["objects"]

        metadata_key = "pcd_metadata"
        if metadata_key not in target.keys():
            metadata_key = "radar_pcd_metadata"

        # fields, dtypes, and np dtypes to construct numpy recarray from list of points from json file for each object
        fields = target[metadata_key]["fields"]
        dtypes = target[metadata_key]["dtypes"]

        np_dtype = np.dtype(
            {
                "names": ast.literal_eval(fields),
                "formats": ast.literal_eval(dtypes),
            }
        )

        labels: List[torch.Tensor] = []
        points: List[np.recarray] = []

        assert anns, f"empty annotations received {target_path}"

        for i in anns:
            class_id = metadata.INFRA3DRC_TO_CLASS_ID_REMAP_WITHOUT_GROUP[
                str(i["category_id"]).zfill(2)
            ]  # class id after removing child, truck, and group.
            labels.append(int(class_id))
            points_list = list(map(tuple, i["points"]))
            assert i["points"]

            # print(pcd_path)
            points_recarray = np.rec.array(points_list, np_dtype)
            points.append(points_recarray)

        annotation_data["name"] = pcd_path.name
        annotation_data["date_captured"] = target[metadata_key]["date_captured"]
        # pcd_id must be taken care when the entire dataset is ready. each sample must have a unique id.
        pcd_id = idx if idx is not None else 0
        annotation_data["pcd_id"] = torch.tensor([pcd_id])
        annotation_data["labels"] = torch.tensor(labels)
        annotation_data["points"] = points

        # handling the case when either static or dynamic cloud has zero points
        if pcd_cloud_dynamic.size == 0:
            raw_cloud = pcd_cloud_static
        elif pcd_cloud_static.size == 0:
            raw_cloud = pcd_cloud_dynamic
        else:
            raw_cloud = np.append(pcd_cloud_dynamic, pcd_cloud_static)
            raw_cloud = raw_cloud.view(np.recarray)

        # sort the raw cloud ising index field. Alternatively we can also shuffle the cloud.
        raw_cloud.sort(order="index")

        if self.field_names is not None:
            # extract fields of interest.
            return_cloud = rfn.repack_fields(raw_cloud[self.field_names])

        # convert recarray to ndarray and then torch tensor.
        return_cloud_tensor = torch.from_numpy(
            rfn.structured_to_unstructured(return_cloud)
        )

        return return_cloud_tensor, annotation_data, raw_cloud #, all_points

    def __len__(self) -> int:
        return len(self.pcds)

    @property
    def name(self):
        print("3D Radar Dataset")
