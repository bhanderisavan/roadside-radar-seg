import argparse
import pickle
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch

from roadside_radar_seg.configs import get_cfg
from roadside_radar_seg.definitions import PROJECT_ROOT
from roadside_radar_seg.engine import DefaultTrainer
from roadside_radar_seg.utils import get_model_summary
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42) 

if __name__ == "__main__":
    conf_parser = argparse.ArgumentParser(description="train radar detector", add_help=False)
    conf_parser.add_argument(
        "--config", 
        required=True, 
        help="relative path to the yaml config file."
    )

    temp_args, _ = conf_parser.parse_known_args()

    # default config
    cfg = get_cfg()

    config_path = Path(PROJECT_ROOT).joinpath(temp_args.config)
    if not config_path.is_file():
        raise ValueError(
            "Unknown config path, please provide config path relative to the project."
        )

    # set the cfg parameters from yaml config file
    cfg.merge_from_file(str(config_path))

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=temp_args.config, help="path to yaml")
    parser.add_argument("--checkpoint", type=int, default=cfg.SOLVER.CHECKPOINT_PERIOD, help="saving interval")
    parser.add_argument("--batch", type=int, default=cfg.DATALOADER.BATCH_SIZE, help="batch size")
    parser.add_argument("--epochs", type=int, default=cfg.SOLVER.MAX_EPOCHS, help="maximum epochs to train")
    parser.add_argument("--device", type=str, default=cfg.MODEL.DEVICE, choices=["cpu", "cuda"], help="Inference device")
    
    args = parser.parse_args()

    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint
    cfg.DATALOADER.BATCH_SIZE = args.batch
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    cfg.MODEL.DEVICE = args.device

    if cfg.MODEL.DEVICE == "cuda":
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda:0"
            torch.cuda.set_device(torch.device(cfg.MODEL.DEVICE))
        else:
            print("WARNING: CUDA requested but no GPU found. Falling back to CPU.")
            cfg.MODEL.DEVICE = "cpu"

    output_dir = Path(__file__).parent.joinpath(
        "results","experiments", 
        f"{str(datetime.today().strftime('%Y_%m_%d-%H_%M_%S'))}__{config_path.stem}",
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()

    trainer = DefaultTrainer(cfg)

    if Path(cfg.MODEL.WEIGHTS).is_file():
        trainer.load_checkpoint()
        trainer.freeze_semseg_weights()

    model_summary = get_model_summary(trainer.model)

    pprint(model_summary)
    pprint(cfg)
    # save model summary
    with open(str(Path(cfg.OUTPUT_DIR).joinpath("model_summary.txt")), "w") as f:
        f.write(model_summary.get_string())

    if Path(cfg.MODEL.WEIGHTS).is_file():
        print("loading checkpoint")
        trainer.load_previous_checkpoint(cfg.MODEL.WEIGHTS)

    cfg_to_save = deepcopy(cfg)
    del cfg_to_save["SOLVER"]
    del cfg_to_save["ROS_DATA_READER"]
    cfg_to_save.freeze()
    # save the config pickle file
    with open(str(Path(cfg.OUTPUT_DIR).joinpath("config.pkl")), "wb") as f:
        pickle.dump(cfg_to_save, f)

    trainer.train()
