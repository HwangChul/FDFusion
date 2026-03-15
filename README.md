# FDFusion: Efficient text-guided infrared-visible image fusion via fine-tuned lightweight VLM and Dual-branch feature modeling
Code and dataset for ***FDFusion (Infrared Physics & Technology 2026).***

## Information

- [*[Project]*](https://github.com/HwangChul/FDFusion)  
- [*[Paper]*](https://www.sciencedirect.com/science/article/pii/S1350449526001349)  
- [*[Dataset]*](https://drive.google.com/file/d/1Upbnds2mUWW_DxsNWZb_wEPmM7njgT-e/view?usp=drive_link)


## Abstract

Infrared-visible image fusion plays a crucial role in enhancing scene perception in complex environments. Recent work has introduced large-scale Vision Language Models (VLMs) into the fusion pipeline to provide semantic priors. However, this reliance leads to substantial computational overhead. Furthermore, such approaches still exhibit notable shortcomings in preserving fine-grained textural details and effectively integrating high-level semantic information. To address these challenges, we propose **FDFusion**, an efficient text-guided infrared-visible image fusion framework based on a **F**ine-tuned lightweight VLM and **D**ual-branch feature modeling. Specifically, FDFusion employs a fine-tuned lightweight VLM to generate high-quality textual descriptions as prior information for guiding modal feature integration. It further incorporates a texture-semantic dual-branch modeling strategy to achieve more targeted semantic guidance. Additionally, we design a cross-layer attention module that establishes explicit alignment and selection mechanisms between features at different semantic depths. Experiments across multiple public datasets demonstrate that FDFusion outperforms state-of-the-art methods in information richness, texture preservation, gradient detail, and semantic consistency. Efficiency analysis further reveals that our fine-tuned lightweight model exhibits significant advantages in GPU memory consumption and inference latency compared to strategies relying on large-scale VLMs, while maintaining competitive fusion quality.


### Network Architecture

<img src="img\Framework.png" width="80%" align=center />

Our FDFusion is implemented in ``net/FDFusion.py``.

### Virtual Environment
```
conda create -n fdfusion python=3.9
conda activate fdfusion
pip install -r requirements.txt
```

### Data Preparation

Download the processed dataset provided in our paper from [this link](https://drive.google.com/file/d/1Upbnds2mUWW_DxsNWZb_wEPmM7njgT-e/view?usp=drive_link). The corresponding DataLoader is in ``utils/H5_read.py``.

### Running

Please run 
```
python test.py
``` 
to perform image fusion. The output fusion results will be saved in the ``'./output/MSRS'``  folder.

### Results
**Quantitative evaluation**

| Methods | EN | SD | SF | AG | VIF | $Q^{AB/F}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LFDT | 6.65 | 43.05 | 11.23 | 3.63 | 1.02 | 0.69 |
| CDDFuse | 6.70 | **43.39** | 11.56 | 3.74 | 1.05 | 0.69 |
| LRRNet | 6.19 | 31.78 | 8.46 | 2.63 | 0.54 | 0.46 |
| DDBF | 5.97 | 28.42 | 8.55 | 2.78 | 0.63 | 0.58 |
| CFNet | 6.63 | 42.23 | 10.30 | 3.40 | 0.68 | 0.50 |
| SDCFusion | 6.72 | 42.66 | 11.83 | 3.94 | 1.03 | 0.70 |
| FILM | 6.72 | 43.17 | 11.70 | 3.84 | **1.06** | **0.73** |
| Our | **6.73** | 43.22 | **11.95** | **3.96** | 1.04 | 0.72 |

**Qualitative evaluation**

<img src="img\MSRS.png" width="80%" align=center />