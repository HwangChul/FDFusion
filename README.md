# FDFusion: Efficient text-guided infrared-visible image fusion via fine-tuned lightweight VLM and Dual-branch feature modeling
Code and dataset for ***FDFusion.***

## Information

- [*[Project]*](https://github.com/HwangChul/FDFusion)  
- [*[Paper]*](https://www.sciencedirect.com/science/article/pii/S1350449526001349)  
- [*[Dataset]*](https://drive.google.com/file/d/1Upbnds2mUWW_DxsNWZb_wEPmM7njgT-e/view?usp=drive_link)


## Abstract

Infrared-visible image fusion plays a crucial role in enhancing scene perception in complex environments. Recent work has introduced large-scale Vision Language Models (VLMs) into the fusion pipeline to provide semantic priors. However, this reliance leads to substantial computational overhead. Furthermore, such approaches still exhibit notable shortcomings in preserving fine-grained textural details and effectively integrating high-level semantic information. To address these challenges, we propose **FDFusion**, an efficient text-guided infrared-visible image fusion framework based on a **F**ine-tuned lightweight VLM and **D**ual-branch feature modeling. Specifically, FDFusion employs a fine-tuned lightweight VLM to generate high-quality textual descriptions as prior information for guiding modal feature integration. It further incorporates a texture-semantic dual-branch modeling strategy to achieve more targeted semantic guidance. Additionally, we design a cross-layer attention module that establishes explicit alignment and selection mechanisms between features at different semantic depths. Experiments across multiple public datasets demonstrate that FDFusion outperforms state-of-the-art methods in information richness, texture preservation, gradient detail, and semantic consistency. Efficiency analysis further reveals that our fine-tuned lightweight model exhibits significant advantages in GPU memory consumption and inference latency compared to strategies relying on large-scale VLMs, while maintaining competitive fusion quality.


### Network Architecture

<img src="img\Framework.png" width="70%" align=center />

Our FDFusion is implemented in ``net/FDFusion.py``.

### Running

Download the processed dataset provided in our paper from [this link](https://drive.google.com/file/d/1Upbnds2mUWW_DxsNWZb_wEPmM7njgT-e/view?usp=drive_link). The corresponding DataLoader is in ``utils/H5_read.py``.

Please run 
```
python test.py
``` 
to perform image fusion. The output fusion results will be saved in the ``'./output/MSRS'``  folder.
