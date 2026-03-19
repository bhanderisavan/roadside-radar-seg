<!-- title: RoadsideRadar Dataset -->

<div align="center">

# RoadsideRadar Dataset

<u>[Savankumar Bhanderi](https://www.linkedin.com/in/savankumarbhanderi)</u><sup>1,<a href="mailto:savankumar.bhanderi@thi.de?subject=RoadsideRadar" style="color: #4799e0; text-decoration: underline;">📧</a></sup>,&nbsp; <u>[Shiva Agrawal](https://www.linkedin.com/in/shiva-agrawal-06562510a/)</u><sup>1</sup>,&nbsp; <u>[Gordon Elger](https://www.linkedin.com/in/gordon-elger-48a658/)</u><sup>1</sup>

<sup>1</sup><u>[Technische Hochschule Ingolstadt](https://www.thi.de/en/research/institute-of-innovative-mobility-iimo/research-areas/sensor-technology-and-sensor-data-fusion/)</u>

<i><span style="color: black;">Springer Nature Scientific Reports 2025 </span></i>

[![paper](https://img.shields.io/badge/Paper-Nature%20Scientific%20Reports-800000?logo=openaccess&logoColor=white)](https://www.nature.com/articles/s41598-025-23019-6)&nbsp;&nbsp; 
[![repo](https://img.shields.io/badge/Github-Codebase-10b981?logo=github)](https://github.com/bhanderisavan/roadside-radar-seg)&nbsp;&nbsp;
[![repo](https://img.shields.io/badge/Thesis-THI-b7791f?logo=gitbook&logoColor=white)](https://opus4.kobv.de/opus4-haw/frontdoor/index/index/searchtype/simple/query/%2A%3A%2A/browsing/true/doctypefq/masterthesis/docId/5145/start/8/rows/50)


<p align="center">
<img  src="images/dataset_with_setup.png" width="90%" />
</p>

</div>

<div style="text-align: justify">

## Introduction

The [RoadsideRadar Dataset](https://www.nature.com/articles/s41598-025-23019-6) contains recordings from a 3+1D automotive Radar sensor, which was mounted on one intelligent roadside measurement setup, shown in the left side of above figure[[1]](https://ieeexplore.ieee.org/abstract/document/10209570). The synchronized  anonymized camera images are also added for visualization pupose. This dataset accompanies the paper [*Deep segmentation of 3+1D radar point cloud for real-time roadside traffic user detection*](https://www.nature.com/articles/s41598-025-23019-6) and the master's thesis [*Real-time semantic and instance segmentation of 3D radar point cloud for smart infrastructure-based road user detection.*](https://opus4.kobv.de/opus4-haw/frontdoor/index/index/searchtype/simple/query/%2A%3A%2A/browsing/true/doctypefq/masterthesis/docId/5145/start/8/rows/50)

The dataset contains 5399 frames of radar point clouds. In addition to the point cloud data from the radar sensor, semantic and instance annotations on a point-wise level from 5 different classes are provided. 

This dataset supports research in:
-   Roadside Radar-based perception
-   Semantic segmentation of radar point clouds
-   Instance segmentation of radar point clouds



## Overview
- [RoadsideRadar Dataset](#roadsideradar-dataset)
  - [Introduction](#introduction)
  - [Overview](#overview)
  - [Dataset Structure](#dataset-structure)
  - [Labeling Information](#labeling-information)
  - [Naming Convention](#naming-convention)
  - [Dataset Statistics](#dataset-statistics)
  - [Radar Point Cloud Fields](#radar-point-cloud-fields)
  - [Annotation Format](#annotation-format)
  - [License](#license)
  - [Funding](#funding)
  - [Citation](#citation)
  - [Contact Us](#contact-us)

---


## Dataset Structure

The dataset is provided in train, val, and test splits. Each split includes radar pcds, corresponding synchronized and anonymized camera images, and the annotation jsons. 

```---
RoadsideRadar
├─────sensor.json               # contains information about the sensor used.
├─────README.md
└─────data
      ├───bg_sub_grids/         # contains .npy background subtraction grid.
      └───splits
          ├───train
          │   ├───images/       # contains camera .png images.
          │   ├───pcds/         # contains radar .pcd files.
          │   └───annotations/  # contains .json annotaion files.
          ├───val
          │   ├───images/
          │   ├───pcds/
          │   └───annotations/
          └───test
              ├───images/
              ├───pcds/
              └───annotations/
```
---

## Labeling Information
This dataset is an extended version of the [INFA-3DRC dataset](https://github.com/FraunhoferIVI/INFRA-3DRC-Dataset), with an increased number of frames and a subset of class categories. Specifically, the following modifications are made to the INFRA-3DRC dataset:

- The scenes with parking spots are removed.
- The Radar frames with object category group are removed.
- Further frames of adverse weather and poor lighting conditions are added.

Please refer to this [manuscript](https://ieeexplore.ieee.org/abstract/document/10459049) for labeling information of the INFRA-3DRC dataset. The additional frames were labeled manually.  
 
## Naming Convention

<div style="text-align: justify">

Note that this dataset is mainly intended for radar focus research only. As such, the name of the camera images do not resemble the name of the radar frames. However, the names of the radar pcd and the corresponding annotation json file are the same. Moreover, each json annotation file contains information about the corresponsing synchronized image. The synchronized image for any given frame can be obtained using the annoatation json file ```(annotation_dict["cam_image"]["file_name"])```. Also note that software based [time-synchronization](https://ieeexplore.ieee.org/abstract/document/10459049) was used for obtaining the radar-camera pairs, leading to difference in the time stamps of radar and camera files. The filename include timestamps in the format ```YYYY-MM-DD-HH-MM-SS-MSEC```. Additionally, the radar frames also include the index of the background subtraction grid at the end ```*_bg{idx}.pcd```.

</div>

```
radar_01__2023-06-02-21-23-51-372_bg0.pcd # radar pcd
camera_01__2023-06-02-21-23-51-376.png    # camera image      
radar__2023-06-02-21-23-51-372_bg0.json   # annotation json
```
---

## Dataset Statistics

The dataset contains total of 5399 frames of radar point clouds. The training, val, and test splits contain 3780, 810, and 809 frames respectively. Details about the class distribution of the labeled objects in each split are given in below table. 
<div align="center">
  <h3>📊 Dataset Statistics & Class Distribution</h3>
  <p><i>A comprehensive breakdown of radar points and objects across the Train, Val, and Test splits.</i></p>

<table style="width: 100%; max-width: 1000px; border-collapse: collapse; margin-left: auto; margin-right: auto;">
  <thead>
    <tr style="border-top: 2px solid #333;">
      <th rowspan="2" style="border-bottom: 1px solid #dfe2e5;"><div align="center">Class</div></th>
      <th colspan="3" style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #2e7d32; padding-bottom: 5px; margin: 0 10px;">Total Points</div>
      </th>
      <th colspan="3" style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #4c51bf; padding-bottom: 5px; margin: 0 10px;">Total Objects</div>
      </th>
      <th colspan="3" style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #b7791f; padding-bottom: 5px; margin: 0 10px;">Avg. Points / Object</div>
      </th>
    </tr>
    <tr>
      <th><div align="center">Train</div></th><th><div align="center">Val</div></th><th><div align="center">Test</div></th>
      <th><div align="center">Train</div></th><th><div align="center">Val</div></th><th><div align="center">Test</div></th>
      <th><div align="center">Train</div></th><th><div align="center">Val</div></th><th><div align="center">Test</div></th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>Person</code></td><td align="center">4297</td><td align="center">896</td><td align="center">889</td><td align="center">1920</td><td align="center">408</td><td align="center">392</td><td align="center">2.24</td><td align="center">2.20</td><td align="center">2.26</td></tr>
    <tr><td><code>Bicycle</code></td><td align="center">6517</td><td align="center">1403</td><td align="center">1274</td><td align="center">2723</td><td align="center">578</td><td align="center">544</td><td align="center">2.40</td><td align="center">2.42</td><td align="center">2.35</td></tr>
    <tr><td><code>Motorcycle</code></td><td align="center">679</td><td align="center">140</td><td align="center">179</td><td align="center">214</td><td align="center">47</td><td align="center">53</td><td align="center">3.17</td><td align="center">2.97</td><td align="center">3.38</td></tr>
    <tr><td><code>Car</code></td><td align="center">15964</td><td align="center">3337</td><td align="center">3612</td><td align="center">3786</td><td align="center">785</td><td align="center">880</td><td align="center">4.22</td><td align="center">4.25</td><td align="center">4.10</td></tr>
    <tr><td><code>Bus</code></td><td align="center">12483</td><td align="center">2420</td><td align="center">2632</td><td align="center">680</td><td align="center">145</td><td align="center">137</td><td align="center">18.35</td><td align="center">16.69</td><td align="center">19.21</td></tr>
    <tr><td><code>Background</code></td><td align="center">110909</td><td align="center">26746</td><td align="center">25507</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
    <tr style="border-top: 1.5px solid #333; border-bottom: 2px solid #333;">
    <tr style="border-top: 1.5px solid #333; border-bottom: 2px solid #333;">
      <td><b>Total (Σ)</b></td>
      <td align="center"><b>150939</b></td><td align="center"><b>34093</b></td><td align="center"><b>34942</b></td>
      <td align="center"><b>9323</b></td><td align="center"><b>1963</b></td><td align="center"><b>2006</b></td>
      <td align="center">—</td><td align="center">—</td><td align="center">—</td>
    </tr>
  </tbody>
</table>

</div>

</br>

Further statistics of the number of static and dynamic points per class in the dataset is provided in below table.

<div align="center">
  <h3>📊 Static vs. Dynamic Point Distribution</h3>
  <p><i>Class-wise static and dynamic radar points distribution</i></p>

<table style="width: 100%; max-width: 700px; border-collapse: collapse; margin-left: auto; margin-right: auto;">
  <thead>
    <tr style="border-top: 2px solid #333;">
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 1px solid #dfe2e5; padding-bottom: 5px; margin: 0 5px;">Category</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #2e7d32; padding-bottom: 5px; margin: 0 5px;">Person</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #4c51bf; padding-bottom: 5px; margin: 0 5px;">Bicycle</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #b7791f; padding-bottom: 5px; margin: 0 5px;">Motorcycle</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #9b2c2c; padding-bottom: 5px; margin: 0 5px;">Car</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #553c9a; padding-bottom: 5px; margin: 0 5px;">Bus</div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px 10px;"><code>Static points</code></td>
      <td align="center">285</td>
      <td align="center">638</td>
      <td align="center">78</td>
      <td align="center">2370</td>
      <td align="center">0</td>
    </tr>
    <tr>
    <tr style="border-top: 1.5px solid #333; border-bottom: 2px solid #333;">
      <td style="padding: 8px 10px;"><code>Dynamic points</code></td>
      <td align="center">5797</td>
      <td align="center">8556</td>
      <td align="center">920</td>
      <td align="center">20543</td>
      <td align="center">17535</td>
    </tr>
    <tr style="border-top: 1.5px solid #333; border-bottom: 2px solid #333;">
      <td style="padding: 10px;"><b>Total (Σ)</b></td>
      <td align="center"><b>6082</b></td>
      <td align="center"><b>9194</b></td>
      <td align="center"><b>998</b></td>
      <td align="center"><b>22913</b></td>
      <td align="center"><b>17535</b></td>
    </tr>
  </tbody>
</table>

</div>

</br>

For more information, the readers are requested to explore our [paper](https://www.nature.com/articles/s41598-025-23019-6), and this [master's thesis](https://opus4.kobv.de/opus4-haw/frontdoor/index/index/searchtype/simple/query/%2A%3A%2A/browsing/true/doctypefq/masterthesis/docId/5145/start/8/rows/50).

--- 
## Radar Point Cloud Fields

Each radar frame is stored as a .pcd (**p**oint **c**loud **d**ata) file, and contains information about the radar points in that frame. Each radar point contains the following fields:

<div align="center">
  <h3>📑 Radar Point Attribute Definitions</h3>
  <p><i>Specifications of the raw and processed features provided for each radar point (detection).</i></p>

<table style="width: 100%; max-width: 1000px; border-collapse: collapse; margin-left: auto; margin-right: auto;">
  <thead>
    <tr style="border-top: 2px solid #333;">
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #2e7d32; padding-bottom: 5px; margin: 0 5px;">Parameter</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #4c51bf; padding-bottom: 5px; margin: 0 5px;">Unit</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #b7791f; padding-bottom: 5px; margin: 0 5px;">Dtype</div>
      </th>
      <th style="border-bottom: none !important;">
        <div align="center" style="border-bottom: 2.5px solid #9b2c2c; padding-bottom: 5px; margin: 0 5px;">Definition</div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>index</code></td>
      <td align="center">no unit</td>
      <td align="center"><code>uint16</code></td>
      <td style="padding: 10px;">Unique index given to each radar point in one frame.</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>range</code></td>
      <td align="center">meter</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Direct distance of the radar point with respect to origin of the radar sensor (Polar).</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>azimuth_angle</code></td>
      <td align="center">radians</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Horizontal angle with respect to sensor origin (Polar).</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>elevation_angle</code></td>
      <td align="center">radians</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Vertical angle with respect to sensor origin (Polar).</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>range_rate</code></td>
      <td align="center">m/sec</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Doppler speed. <b>Negative</b>: approaching; <b>Positive</b>: receding.</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>rcs</code></td>
      <td align="center">dBsm</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Radar Cross Section (signal strength).</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>x, y, z</code></td>
      <td align="center">meter</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Longitudinal, lateral, and vertical distance from sensor origin (Cartesian).</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>x_ground, y_ground, z_ground</code></td>
      <td align="center">meter</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Longitudinal, lateral, and vertical distance from the <b>ground</b> origin (Cartesian).</td>
    </tr>
    <tr>
      <td style="padding: 10px; background-color: #fcfcfc;"><code>u, v</code></td>
      <td align="center">pixels</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Projected pixel coordinates in the 2D image plane.</td>
    </tr>
    <tr style="border-bottom: 2px solid #333;">
      <td style="padding: 10px; background-color: #fcfcfc;"><code>v_x, v_y</code></td>
      <td align="center">m/sec</td>
      <td align="center"><code>float32</code></td>
      <td style="padding: 10px;">Longitudinal and lateral velocity components of the radial velocity.</td>
    </tr>
  </tbody>
</table>

</div>

---
## Annotation Format

Annotations are stored in JSON format  and provide point-level semantic as well as object-level instance labels for radar points. Each object contains radar points belonging to that instance. Note that there are annotations present for image (i.e. bouning box, segmentation mask). The class definitions are given below.

<div align="center">
  <h3>🆔 Category ID to class name  </h3>


<table style="width: 100%; max-width: 500px; border-collapse: collapse; margin-left: auto; margin-right: auto;">
  <thead>
    <tr style="border-top: 2px solid #333;">
      <th style="border-bottom: 1px solid #dfe2e5; padding: 10px; background-color: #fcfcfc;">
        <div align="center">Category ID</div>
      </th>
      <th style="border-bottom: none !important;"><div align="center" style="border-bottom: 2.5px solid #2e7d32; padding-bottom: 5px; margin: 0 5px;">1</div></th>
      <th style="border-bottom: none !important;"><div align="center" style="border-bottom: 2.5px solid #4c51bf; padding-bottom: 5px; margin: 0 5px;">4</div></th>
      <th style="border-bottom: none !important;"><div align="center" style="border-bottom: 2.5px solid #b7791f; padding-bottom: 5px; margin: 0 5px;">5</div></th>
      <th style="border-bottom: none !important;"><div align="center" style="border-bottom: 2.5px solid #9b2c2c; padding-bottom: 5px; margin: 0 5px;">6</div></th>
      <th style="border-bottom: none !important;"><div align="center" style="border-bottom: 2.5px solid #553c9a; padding-bottom: 5px; margin: 0 5px;">7</div></th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 2px solid #333;">
      <td align="center" style="padding: 15px; background-color: #fcfcfc;"><b>Class Name</b></td>
      <td align="center" style="font-weight: 500;">Adult</td>
      <td align="center" style="font-weight: 500;">Bicycle</td>
      <td align="center" style="font-weight: 500;">Motorcycle</td>
      <td align="center" style="font-weight: 500;">Car</td>
      <td align="center" style="font-weight: 500;">Bus</td>
    </tr>
  </tbody>
</table>

</div>
 
</br>

Below is an examplary annotation JSON structure.

``` json
{
  "info": {
    "description": "RoadsideRadar_Dataset",
    "year": "2025",
    "version": "1"
  }, 
  "objects": [
    {
      "category_id": 7,
      "points": [
        [101, 54.9962, -0.4125, ...],
        [102, 55.9972, -0.4575, ...],
      ]
    }
  ],
  "categories": [
    {
      "category_id": "7",
      "supercategory": "vehicle",
      "name": "bus",
    }
  ],
  "cam_image": {
    "file_name": "camera_01__2023-06-02-21-23-51-376.png",
    "height": 1216,
    "width": 1920
  },
  "pcd_metadata": {
    "pcd_name": "radar_01__2023-06-02-21-23-51-372.pcd",
    "points": 505,
    "fields": "['index', 'range', 'azimuth_angle',  ...]",
    "dtypes": "['uint16', 'float32', 'float32', ...]",
  }
}
```
---
## License

The data set is licensed under Creative Commons Attribution Non Commercial Share Alike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). Hence, the data set must not be used for any commercial use cases.

---
## Funding

This work was supported by the Bavarian Ministry of Economic Affairs, Regional Development and Energy (StMWi), Germany within the Project “InFra — Intelligent Infrastructure.”

## Citation

If this dataset has contributed to your work, we would appreciate citing our [paper](https://www.nature.com/articles/s41598-025-23019-6) and giving the [Github](https://github.com/bhanderisavan/roadside-radar-seg) repository a star.

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

@dataset{bhanderi_2025_19056521,
author={Bhanderi, Savankumar andAgrawal, Shiva and Elger, Gordon},
title={RoadsideRadar: A Roadside 3+1D Automotive RadarPoint Cloud Dataset for Semantic and Instance Segmentation},
month=march,
year=2025,
publisher={Zenodo},
version={1.0},
doi={10.5281/zenodo.19056521},
url={https://doi.org/10.5281/zenodo.19056521},
}
```

---
## Contact Us

For questions regarding the dataset or collabotations, please contact the dataset authors.

- [Savankumar Bhanderi](https://www.linkedin.com/in/savankumarbhanderi)<sup><a href="mailto:savankumar.bhanderi@thi.de?subject=RoadsideRadar" style="color: #4799e0; text-decoration: underline;">📧</a></sup>  

</div>