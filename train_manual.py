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
torch.manual_seed(42)  # for reproducibility.

cfg = get_cfg()
config_name = ""
# set the cfg parameters from yaml config file
cfg.merge_from_file(str(Path(PROJECT_ROOT).joinpath("experiments", config_name)))

cfg.TRAIN.USE_BGSUB = True
cfg.VAL.USE_BGSUB = True

cfg.TRAIN.DATASET_PATH = ("")
cfg.VAL.DATASET_PATH = ("")

if cfg.MODEL.DEVICE != "cpu":
    torch.cuda.set_device(torch.device(cfg.MODEL.DEVICE))

output_dir = Path(__file__).parent.joinpath(
    "results", "experiments", 
    f"{str(datetime.today().strftime('%Y_%m_%d-%H_%M_%S'))}__{config_name.split('.')[0]}",
)
output_dir.mkdir(exist_ok=True, parents=True)
cfg.OUTPUT_DIR = str(output_dir)
cfg.freeze()

trainer = DefaultTrainer(cfg)
model_summary = get_model_summary(trainer.model)
pprint(model_summary)
print(trainer.model)

param_size = 0
for param in trainer.model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in trainer.model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
#exit(1)
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
