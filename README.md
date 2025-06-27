# E-BayesSAM
This repository provides the official implementation of our MICCAI 2025 accepted paper:  
**E-BayesSAM: Efficient Bayesian Adaptation of SAM with Self-Optimizing KAN-Based Interpretation for Uncertainty-Aware Ultrasonic Segmentation**.

---

## 🔧 Getting Started

### Inference
To perform segmentation on a sample image, run:
```bash
python inference_demo.py -p data_demo/images/amos_0004_75.png
```

### Inference with Uncertainty Visualization
To visualize both the segmentation and its uncertainty map, run:
```bash
python inference_demo.py -p data_demo/images/amos_0004_75.png -u True
```
---

## 🙏 Acknowledgements

We sincerely thank the authors of [**SAM-Med2D**](https://github.com/OpenGVLab/SAM-Med2D) and [**MedSAM**](https://github.com/bowang-lab/MedSAM) for their open-source contributions, which laid the foundation for our development.
