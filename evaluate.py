import argparse
import json
import pickle
from pathlib import Path
from time import time

import numpy.lib.recfunctions as rfn
import torch
from pycm import ConfusionMatrix
from tqdm import tqdm

from roadside_radar_seg.data import build_val_loader, metadata
from roadside_radar_seg.definitions import PROJECT_ROOT
from roadside_radar_seg.evaluation import build_cm_evaluator, build_map_evaluator
from roadside_radar_seg.models import build_model


# Helper Functions
def update_cm_evaluator(cm_evaluator, pred_labels: torch.Tensor, gt_labels: torch.Tensor):
    if pred_labels.ndim == 3:
        pred_labels = pred_labels.squeeze(-1)
    gt_labels = gt_labels.to(pred_labels)
    valid_points_mask = pred_labels != -1
    gt_labels = gt_labels[valid_points_mask]
    pred_labels = pred_labels[valid_points_mask]
    cm_evaluator.update(preds=pred_labels, target=gt_labels)

def update_map_evaluator(evaluator, radar_cluster_list, gt_target_list, device):
    gt_lookup = {gt['name']: gt for gt in gt_target_list}
    aligned_preds = []
    aligned_gts = []

    for pred_frame in radar_cluster_list:
        frame_name = pred_frame.frame_name 
        if frame_name not in gt_lookup:
            continue
            
        gt = gt_lookup[frame_name]
        current_res_dict = {"clusters": [], "labels": [], "scores": []}
        for cluster in pred_frame:
            cluster_xyz = cluster.radar_points[["x", "y", "z"]]
            cluster_tensor = torch.from_numpy(
                rfn.structured_to_unstructured(cluster_xyz).copy()
            ).to(device)
            current_res_dict["clusters"].append(cluster_tensor)
            current_res_dict["labels"].append(cluster.category.value)
            current_res_dict["scores"].append(cluster.category_confidence)

        current_res_dict["labels"] = torch.tensor(current_res_dict["labels"], device=device)
        current_res_dict["scores"] = torch.tensor(current_res_dict["scores"], device=device)

        current_gt_dict = {"clusters": [], "labels": gt["labels"].to(device)}
        for i in gt["points"]:
            gt_cluster_xyz = i[["x", "y", "z"]]
            gt_cluster_tensor = torch.from_numpy(
                rfn.structured_to_unstructured(gt_cluster_xyz).copy()
            ).to(device)
            current_gt_dict["clusters"].append(gt_cluster_tensor)

        aligned_preds.append(current_res_dict)
        aligned_gts.append(current_gt_dict)

    evaluator.update(preds=aligned_preds, target=aligned_gts)

def get_epoch_stats_dict(map_dict, cm):
    cls_wise_map = {
        metadata.CLASS_ID_TO_NAMES_WITHOUT_GROUP[str(cls_idx.item()).zfill(2)]: round(
            map_dict["map_per_class"][i].item(), 4
        ) for i, cls_idx in enumerate(map_dict["classes"])
    }
    stats = {
        "mAP_50": round(map_dict["map_50"].item(), 4),
        "mAP": map_dict["map"].item(),
        "mAP_classwise": cls_wise_map,
        "mAP_75":map_dict["map_75"].item(),
        "cm": cm.cpu().tolist(),
    }

    if map_dict.__contains__("map_30"):
        stats["mAP_30"] = round(map_dict["map_30"].item(), 4)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Radar Detection Evaluation Tool")
    parser.add_argument("--data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--bg_sub_grid_folder", type=str, required=True, help="Path to background subtraction grid folder")
    parser.add_argument("--ckpt", type=str, required=True, help="Absolute path to trained model .pth file")
    parser.add_argument("--config", type=str, required=True, help="Absolute path to the config.pkl")
    parser.add_argument("--out", type=str, default=str(Path(PROJECT_ROOT).joinpath("results")), help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size")

    args = parser.parse_args()

    # Paths Setup
    output_dir = Path(args.out)
    test_out_dir = Path(output_dir).joinpath("test")
    test_out_dir.mkdir(exist_ok=True, parents=True)
    
    #cfg_path = str(Path(PROJECT_ROOT).joinpath(args.config)) 
    cfg_path = args.config
    assert Path(cfg_path).is_file(), f"Invalid config.pkl path {cfg_path}"
    # Load Config
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)

    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(args.ckpt)
    cfg.MODEL.DEVICE = args.device
    cfg.VAL.DATASET_PATH = args.data
    cfg.DATALOADER.BATCH_SIZE = args.batch
    cfg.EVALUATION.IOU_THRESHOLDS = [0.75]
    cfg.BGSUB.GRID_FOLDER_PATH = args.bg_sub_grid_folder


    # Load Model
    model = build_model(cfg)
    model.eval()

    checkpoint_data = torch.load(cfg.MODEL.WEIGHTS, map_location=args.device)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    # Evaluators
    loader = build_val_loader(cfg)
    cm_evaluator = build_cm_evaluator(cfg)
    map_evaluator = build_map_evaluator(cfg)

    print()
    
    inference_times = []

    print(f"Starting evaluation on {args.device}...")
    with torch.no_grad():
        for batch_data in tqdm(loader, unit="batch"):
            input_tensor, gt_targets, raw_cloud = batch_data
            
            t1 = time()
            loss_dict, radar_clusters, cls_labels, gt_labels, _ = model(
                input_tensor, raw_cloud, gt_targets
            )
            inference_times.append(time() - t1)

            update_map_evaluator(map_evaluator, radar_clusters, gt_targets, cfg.MODEL.DEVICE)
            update_cm_evaluator(cm_evaluator, cls_labels, gt_labels)

    # Compute & Save Results
    print(f"Mean Inference Time: {sum(inference_times)/len(inference_times):.4f}s")
    
    test_cm = cm_evaluator.compute()
    map_results = map_evaluator.compute()
    
    stats = get_epoch_stats_dict(
         map_results, test_cm
    )

    with open(str(test_out_dir.joinpath("mAP_stats.json")), "w") as f:
        json.dump(stats, f, indent=4)

    # Confusion Matrix Visualization
    cm_obj = ConfusionMatrix(
        matrix=test_cm.cpu().numpy(),
        classes=["background", "person", "bicycle", "motorcycle", "car", "bus"]
    )
    cm_obj.save_stat(str(test_out_dir.joinpath("cm_stats")))
    print(cm_obj)
    print(stats)
    print(f"Evaluation complete. Results saved to {test_out_dir}")

if __name__ == "__main__":
    main()