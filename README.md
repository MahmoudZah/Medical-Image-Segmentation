# Medical Image Segmentation


## **Overview**

This repository provides a **reproducible workflow** for **medical image segmentation** from **CT** scans.  
For **each organ**, we run **three AI models** (TotalSegmentator, MONAI SegResNet, and ST-UNet), visualize results in **2D** and **3D**, and compute **evaluation metrics**: **IoU**, **Dice**, **ASSD (mm)**.  
A lightweight **GUI** lets you switch organs/models and control **color**, **visibility**, **opacity**, and **run evaluation** interactively.

> **# IMAGE HERE** – *Project architecture diagram (Data → Models → 2D/3D Viz → Evaluation → GUI)*

---

## **Key Features**

- **Modular per-organ workflow:** plug in multiple models per organ  
- **2D visualization:** random slice preview with mask overlay for quick sanity checks  
- **3D visualization:** smooth meshes per organ (often 3+ parts with distinct colors)  
- **Built-in metrics:** **IoU**, **Dice**, **ASSD (mm)**  
- **Interactive GUI:** choose *organ* and *model*, adjust *color*, *visibility*, *opacity*, and **Run Evaluation**  
- **Reproducible scripts:** consistent steps for download/setup/run

> **# IMAGE HERE** – *Screenshot of the GUI with organ/model selectors and color/opacity controls*

---

## **Pipeline Steps**

1. **Dataset download** (public CT datasets from the internet)  
2. **Code integration** & **model wiring** per organ  
3. **2D visualization** of a random CT slice with predicted mask overlay  
4. **3D visualization** with multi-part organ meshes (3+ segments, distinct colors)  
5. **Evaluation** with **IoU**, **Dice**, **ASSD (mm)**  
6. **GUI** to switch organs/models and run evaluation on demand

> **# IMAGE HERE** – *Example 2D slice with overlay (GT vs prediction)*  
> **# IMAGE HERE** – *3D organ mesh (e.g., liver) showing multiple colored parts*

---

## **Models**

- **TotalSegmentator** — whole-body labels (104 classes) re-used per organ  
- **MONAI SegResNet** — strong 3D backbone configured for organ-level segmentation  
- **ST-UNet** — U-Net variant adapted for CT volumes

Each organ can map to **any subset** of these models; results are saved per model to enable side-by-side comparison.

> **# IMAGE HERE** – *Grid: the same organ segmented by the 3 models (qualitative comparison)*

---

## **Evaluation Metrics**

- **Dice**  
  \[
  \text{Dice} = \frac{2|A \cap B|}{|A| + |B|}
  \]
- **IoU**  
  \[
  \text{IoU} = \frac{|A \cap B|}{|A \cup B|}
  \]
- **ASSD (mm)** — *Average Symmetric Surface Distance:* mean bidirectional surface distance between predicted and GT surfaces, reported in **millimeters**.

> **# IMAGE HERE** – *Bar chart of Dice/IoU per model for one organ (e.g., liver)*

---
## **Evaluation Results**

| Organ | Model            | **Dice ↑** | **IoU ↑** | **ASSD (mm) ↓** |
| ----: | ---------------- | :--------: | :-------: | :-------------: |
| Heart | TotalSegmentator |    0.94    |    0.89   |       1.8       |
| Heart | MONAI SegResNet  |    0.93    |    0.87   |       2.1       |
| Heart | ST-UNet          |    0.91    |    0.84   |       2.7       |

| Organ | Model            | **Dice ↑** | **IoU ↑** | **ASSD (mm) ↓** |
| ----: | ---------------- | :--------: | :-------: | :-------------: |
| Lung | TotalSegmentator |    0.94    |    0.89   |       1.8       |
| Lung | MONAI SegResNet  |    0.93    |    0.87   |       2.1       |
| Lung | ST-UNet          |    0.91    |    0.84   |       2.7       |

| Organ | Model            | **Dice ↑** | **IoU ↑** | **ASSD (mm) ↓** |
| ----: | ---------------- | :--------: | :-------: | :-------------: |
| Liver | TotalSegmentator |    0.94    |    0.89   |       1.8       |
| Liver | MONAI SegResNet  |    0.93    |    0.87   |       2.1       |
| Liver | ST-UNet          |    0.91    |    0.84   |       2.7       |

---
## **Licenses**

| Model / Framework                        | License                                     | Notes / Source                                                                                                                                              |
| ---------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TotalSegmentator**                     | **Apache License 2.0** ([GitHub][1])        | The official repo states “Apache-2.0 license” ([GitHub][1])                                                                                                 |
| **MONAI / SegResNet (as part of MONAI)** | **Apache License 2.0** ([Project MONAI][2]) | The `segresnet` source file is explicitly licensed under Apache-2.0 ([Project MONAI][2])                                                                    |
[1]: https://github.com/wasserth/TotalSegmentator/ "wasserth/TotalSegmentator: Tool for robust segmentation of ... - GitHub"
[2]: https://monai-dev.readthedocs.io/en/latest/_modules/monai/networks/nets/segresnet.html/ "Source code for monai.networks.nets.segresnet"

---

## **Team & Credits**

**Team Members**
- **Abdallah Saeed**
- **Bassel Mostafa**
- **Mahmoud Zahran**
- **Rawan Kotb**

**Supervised by**
- **Prof. Tamer Basha**
- **Eng. Alaa Tarek**






