
## **Overview**

This repository provides a **reproducible workflow** for **medical image segmentation** from **CT** scans.  
For **each organ**, we run **Two AI models** (TotalSegmentator, and ST-UNet), visualize results in **2D** and **3D**, and compute **evaluation metrics**: **IoU**, **Dice**, **ASSD (mm)**.  
A lightweight **GUI** lets you switch organs/models and control **color**, **visibility**, **opacity**, and **run evaluation** interactively.

---

## **Key Features**

- **Modular per-organ workflow:** plug in multiple models per organ  
- **2D visualization:** random slice preview with mask overlay for quick sanity checks  
- **3D visualization:** smooth meshes per organ (often 3+ parts with distinct colors)  
- **Built-in metrics:** **IoU**, **Dice**, **ASSD (mm)**  
- **Interactive GUI:** choose *organ* and *model*, adjust *color*, *visibility*, *opacity*, and **Run Evaluation**  
- **Reproducible scripts:** consistent steps for download/setup/run

![Image Alt](https://github.com/MahmoudZah/Medical-Image-Segmentation-/blob/main/assets/overview-lung.png?raw=true)

---

## **Pipeline Steps**

1. **Dataset download** (public CT datasets from the internet)  
2. **Code integration** & **model wiring** per organ  
3. **2D visualization** of a random CT slice with predicted mask overlay  
4. **3D visualization** with multi-part organ meshes (3+ segments, distinct colors)  
5. **Evaluation** with **IoU**, **Dice**, **ASSD (mm)**  
6. **GUI** to switch organs/models and run evaluation on demand


![Video Alt](https://github.com/MahmoudZah/Medical-Image-Segmentation-/blob/main/assets/ttlsmgntHeart.gif?raw=true)  ![Video Alt](https://github.com/MahmoudZah/Medical-Image-Segmentation-/blob/main/assets/ttlsgmntLiver.gif?raw=true)
                                                                                                                    

---

## **Models**

- **TotalSegmentator** — whole-body labels (104 classes) re-used per organ   
- **ST-UNet** — U-Net variant adapted for CT volumes

Each organ can map to **any subset** of these models; results are saved per model to enable side-by-side comparison.

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

---
## **Licenses**

| Model / Framework                        | License                                     | Notes / Source                                                                                                                                              |
| ---------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TotalSegmentator**                     | **Apache License 2.0** ([GitHub][1])        | The official repo states “Apache-2.0 license” ([GitHub][1])                                                              
[1]: https://github.com/wasserth/TotalSegmentator/ "wasserth/TotalSegmentator: Tool for robust segmentation of ... - GitHub"

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






