<div align="center">

# Deep Segmentation of 3+1D Radar Point Cloud for Real-Time Roadside Traffic User Detection

<u>[Savankumar Bhanderi](https://www.linkedin.com/in/savankumarbhanderi)</u><sup>1,<a href="mailto:savankumar.bhanderi@thi.de?subject=RoadsideRadar" style="color: #4799e0; text-decoration: underline;">📧</a></sup>,&nbsp; <u>[Shiva Agrawal](https://www.linkedin.com/in/shiva-agrawal-06562510a/)</u><sup>1</sup>,&nbsp; <u>[Gordon Elger](https://www.linkedin.com/in/gordon-elger-48a658/)</u><sup>1</sup>

<sup>1</sup><u>[Technische Hochschule Ingolstadt](https://www.thi.de/en/research/institute-of-innovative-mobility-iimo/research-areas/sensor-technology-and-sensor-data-fusion/)</u>

<i><span style="color: black;">Springer Nature Scientific Reports 2025 </span></i>

[![paper](https://img.shields.io/badge/Paper-Nature%20Scientific%20Reports-800000?logo=openaccess&logoColor=white)](https://www.nature.com/articles/s41598-025-23019-6)&nbsp;&nbsp; 
[![dataset](https://img.shields.io/badge/Dataset-10.5281/zenodo.19056521-10b981)](https://doi.org/10.5281/zenodo.19056521)&nbsp;&nbsp;
[![repo](https://img.shields.io/badge/Thesis-THI-b7791f?logo=gitbook&logoColor=white)](https://opus4.kobv.de/opus4-haw/frontdoor/index/index/searchtype/simple/query/%2A%3A%2A/browsing/true/doctypefq/masterthesis/docId/5145/start/8/rows/50)
</div>

## Introduction
This paper proposes a deep learning-based 3 + 1 D radar point cloud clustering methodology tailored for smart infrastructure-based perception applications. This approach first performs semantic segmentation of the radar point cloud, followed by instance segmentation to generate well-formed clusters with class labels using a deep neural network. It also detects single-point objects that conventional methods often miss. The described approach is developed and experimented using a smart infrastructure-based sensor setup and it performs segmentation of the point cloud in real-time.

<p align="center">
<img  src="docs/images/method_overview.png" width="90%" />
</p>

</div>

---

## Dataset
Along with the deep learning algorithm, the RoadsideRadar dataset is also provided in this paper. Please visit official [zenodo](https://doi.org/10.5281/zenodo.19056521) webpage to access the dataset. The code in this repository expects the same structure of dataset as provided in the zenodo. For more information, please read [dataset.md](docs/dataset.md) 

## Environment Setup
To dependency conflicts, we recommend using a **Conda** virtual environment. This project is tested for *Python 3.8.10*.

### 1. Clone the Repository
Open your terminal and clone the project to your local machine:
```bash
git clone https://github.com/bhanderisavan/roadside-radar-seg.git
cd roadside-radar-seg
```

### 2. Create the Conda Environment
```bash
conda create -n radar_seg python=3.8.10 -y
conda activate radar_seg
```

### 3. Install Dependencies
```bash
conda install pip -y
pip install -r requirements.txt
```
---
## Training

### 1. Model Configuration
For training, create a .yaml config file similar to [config.yaml]("experiments/config.yaml"). Then follow the instructions below. Note that the value of ```--config``` flag is *relative* config path. Also, if the values of other flags are provided, then it will take priority over values in the .yaml config.

For configuration of ```config.yaml``` file, the follwing parameters are most important.

```bash
cfg.BGSUB.GRID_FOLDER_PATH = "/path/to/RoadsideRadar_Dataset/data/bg_sub_grids"
cfg.TRAIN.DATASET_PATH = "/path/to/RoadsideRadar_Dataset/data/splits/train"
cfg.VAL.DATASET_PATH = "/path/to/RoadsideRadar_Dataset/data/splits/val"
cfg.DATALOADER.BATCH_SIZE = <batch_size>
cfg.MODEL.DEVICE = device
```
Configuration of the model architecture (layers, activation functions, normalization type) can be done within the ```cfg.MODEL.X``` key. Here ```X``` is any of ```INPUT_PROCESSION```, ```BACKBONE```, 
```SEGM_HEAD```, or ```INSTANCE_HEAD```. Please have a look at [config.yaml]("experiments/config.yaml") for more configuration options. 

### 2. Training

```bash
cd /path/to/roadside-radar-seg
conda activate radar_seg
# this script runs training from command line. Note thea config path is relative to project root.
python3 train_cli.py --config "experiments/config.yaml" \
  --checkpoint 10 \
  --batch 64 \
  --device "cpu"
```
If you want to run the script from your favorite code editor, use [train_manual.py]("train_manual.py") script. If you use this script, don't forget to fill the ```config_name```, ```cfg.TRAIN.DATASET_NAME```, and ```cfg.VAL.DATASET_NAME```.

The outputs will be stored in the ```results/YYYY_MM_DD-HH_MM_SS__<config_stem>``` directory. Within this directory, the training script will write the following files:
```bash
runs/events.out.* # tensorboard log file
config.pkl   # config file, this is needed for running inference/evaluation in a later step.
model_summary.txt # tabular data containing all model layers and number of trainable parameters 
# weights will be saved for each epoch
model_epoch_<epoch_idx>.pth # model checkpoint for the epoch nr <epoch_id>.
# for debugging, per frame losses will be saved during training and validation for each epoch
per_frame_losses_train_epoch_<eopch_id>.json # per frame training losses for epoch nr <epoch_id>.
per_frame_losses_val_epoch_<eopch_id>.json # per frame validation losses for epoch nr <epoch_id>.
# for each epoch - results (confusion matrix and mAP values) are stored
results_train_epoch_<epoch_id>.json # cummulative json, i.e. epoch 50 containes results of epoch 1,2...,50.
results_val_epoch_<epoch_id>.json
```

---
## Evaluation

To evaluate a trained checkpoint, run the following snippet.

```bash
cd /path/to/roadside-radar-seg
conda activate radar_seg
python3 evaluate.py --data path/to/RoadsideRadar_Dataset/data/splits/test \
  --bg_sub_grid_folder path/to/RoadsideRadar_Dataset/data/bg_sub_grids \
  --ckpt /path/to/model_epoch_<epoch_id>.pth \
  --config /path/to/config.pkl \
  --out /path/to/save_dir \
  --device "cuda" \
  --batch 8 
```
---
## Inference 
To run inference on a pcd file, run the following snippet.

Note that the script expects path to the background subtraction grid only when the pcd is captured from the location where background subtraction is required. This means that if there is *_bg01.pcd* in the name of the pcd, the provde  ```/path/to/RaodsideRadar_Dataset/data/bg_sub_grids/01.npy``` for --bgsub flag.

```bash
cd /path/to/roadside-radar-seg
conda activate radar_seg
python3 inference_from_file.py --pcd path/to/radar/pcd \
  --weights /path/to/model_epoch_<epoch_id>.pth \
  --config /path/to/config.pkl \
  --device "cuda" \
  --bgsub /path/to/bg_sub_grid.npy 
```

---
## License

The data set is licensed under Creative Commons Attribution Non Commercial Share Alike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). Hence, the data set must not be used for any commercial use cases.

---
## Funding

This work was supported by the Bavarian Ministry of Economic Affairs, Regional Development and Energy (StMWi), Germany within the Project “InFra — Intelligent Infrastructure.”

## Citation

If our work has contributed to your research, we would appreciate citing our [paper](https://www.nature.com/articles/s41598-025-23019-6) and giving the [Github](https://github.com/bhanderisavan/roadside-radar-seg) repository a star.

```
@article{bhanderi2025radar,
title={Deep segmentation of 3+1D radar point cloud for real-time roadside traffic user detection},
author={Bhanderi, S. and Agrawal, S. and Elger, G.},
journal={Scientific Reports},
volume={15},
pages={38489},
year={2025},
doi={10.1038/s41598-025-23019-6}
}
```

```
@dataset{bhanderi_2025_19056521,
author={Bhanderi, Savankumar and Agrawal, Shiva and Elger, Gordon},
title={RoadsideRadar: A Roadside 3+1D Automotive RadarPoint Cloud Dataset for Semantic and Instance Segmentation},
month=march,
year=2025,
publisher={Zenodo},
version={1.0},
doi={10.5281/zenodo.19056521},
url={https://doi.org/10.5281/zenodo.19056521},
}
```

## Contact

For questions, collaborations, or dataset/code access updates, please contact:  

**Savankumar Bhanderi**  

- **Email:** [savankumar.bhanderi@thi.de](mailto:savankumar.bhanderi@thi.de)  
- **GitHub:** [https://github.com/bhanderisavan](https://github.com/bhanderisavan)  
- **ORCID:** [https://orcid.org/0000-0001-7257-6736](https://orcid.org/0000-0001-7257-6736)  
- **Google Scholar:** [link](https://scholar.google.com/citations?user=p0775gsAAAAJ&hl=de&authuser=1)
