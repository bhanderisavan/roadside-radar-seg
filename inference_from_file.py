import argparse
import pickle
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from prettytable import ALL, PrettyTable
import torch

from roadside_radar_seg.data import build_radar_bg_subtractor
from roadside_radar_seg.structures import RadarCluster3dList
from roadside_radar_seg.definitions import PROJECT_ROOT
from roadside_radar_seg.engine import DefaultPredictor
from roadside_radar_seg.utils import PcdHelper

def run_inference(pcd_path, config_path, weight_path, bgsub_path, thresh, device):
    
    # Load Configuration
    with open(str(config_path), "rb") as f:
        cfg = pickle.load(f)
    
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(weight_path)
    cfg.MODEL.INSTANCE_HEAD.INPUT_CONFIDENCE_THRESH_TEST = thresh
    cfg.MODEL.DEVICE = device

    # Initialize Predictor
    predictor = DefaultPredictor(cfg)
    # for reading the pcd
    pcd_helper = PcdHelper()

    # Process Radar Cloud
    radar_cloud = pcd_helper.read_radar_pcd(str(pcd_path))
    
    # Filter by range and velocity
    radar_cloud = radar_cloud[radar_cloud["x"] <= cfg.INPUT.MAX_DEPTH]
    radar_cloud = radar_cloud[abs(radar_cloud["range_rate"]) <= cfg.INPUT.MAX_ABS_VELOCITY]

    # Split Dynamic and Static
    radar_cloud_dynamic = radar_cloud[abs(radar_cloud["range_rate"]) >= cfg.INPUT.DYNAMIC_V_THRESHOLD]
    radar_cloud_static = radar_cloud[abs(radar_cloud["range_rate"]) < cfg.INPUT.DYNAMIC_V_THRESHOLD]

    # Background Subtraction 
    if bgsub_path and bgsub_path.is_file():
        bg_subtractor = build_radar_bg_subtractor(cfg)
        bg_sub_grid = np.load(str(bgsub_path))
        radar_cloud_static, _ = bg_subtractor.perform_bg_sub(
            bg_sub_grid, radar_cloud_static
        )
        radar_raw_cloud = np.append(radar_cloud_dynamic, radar_cloud_static)
    else:
        radar_raw_cloud = radar_cloud_dynamic

    # Prepare Tensor for Model
    radar_raw_cloud = radar_raw_cloud.view(np.recarray)
    radar_raw_cloud.sort(order="index")

    return_cloud = rfn.repack_fields(radar_raw_cloud[cfg.INPUT.INPUT_FIELDS])
    return_cloud_tensor = torch.from_numpy(rfn.structured_to_unstructured(return_cloud))

    # Inference
    radar_cluster_list, cls_labels, cls_scores, out_features = predictor(
        radar_raw_cloud=radar_raw_cloud, input_tensor=return_cloud_tensor.float()
    )

    return radar_cluster_list, cls_labels.cpu().numpy(), cls_scores.cpu().numpy(), out_features


def display_radar_clusters(cluster_list: RadarCluster3dList):
    master_table = PrettyTable()
    

    master_table.field_names = ["Cluster", "Class", "Score(%)", "index      x        y        z       vx       vy"]
    master_table.hrules = ALL 

    for i, cluster in enumerate(cluster_list):
        point_rows = []
        
        for pt in cluster.radar_points:
            row_str = f"{int(pt['index']):<4} {pt['x']:>8.2f} {pt['y']:>8.2f} {pt['z']:>8.2f} {pt['v_x']:>8.2f} {pt['v_y']:>8.2f}"
            point_rows.append(row_str)
        
        points_block = "\n".join(point_rows)

        num_points = len(point_rows)
        padding = "\n" * ((num_points - 1) // 2)
        
        raw_score = cluster.category_confidence * 100
        formatted_score = f"{padding}{raw_score:.2f}"
        
        class_name = f"{padding}{cluster.category.name if hasattr(cluster.category, 'name') else cluster.category}"

        master_table.add_row([
            f"{padding}{i+1}",
            class_name,
            formatted_score,
            points_block
        ])

    # Styling
    master_table.align["index      x        y        z       vx       vy"] = "l"
    return master_table


def main():
    parser = argparse.ArgumentParser(description="Radar 3D Detection Inference CLI")
    
    # Paths
    parser.add_argument("--pcd", type=str, required=True, help="Path to input .pcd file")
    parser.add_argument("--config", type=str, required=True, help="Absolute path to the config.pkl")
    parser.add_argument("--weights", type=str, required=True, help="Absolute path to the model.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--bgsub", type=str, default="", help="Path to background subtraction .npy grid")
    
    # Hyperparameters
    parser.add_argument("--thresh", type=float, default=0.6, help="Confidence threshold for detection")

    args = parser.parse_args()

    # inference
    clusters, labels, scores, sem_seg_features = run_inference(
        Path(args.pcd), 
        Path(args.config), 
        Path(args.weights), 
        Path(args.bgsub), 
        args.thresh,
        args.device
    )

    cluster_list = clusters[0]
    cluster_list.frame_name = Path(args.pcd).name

    master_table = display_radar_clusters(cluster_list)
    print(f"\nRADAR FRAME: {cluster_list.frame_name}")
    print(master_table)


if __name__ == "__main__":
    main()

"""
python predict_radar.py \
--pcd "data/test.pcd" \
--weights "checkpoints/best_model.pth" \
--bgsub "data/bg_grid.npy" \
--thresh 0.45
"""