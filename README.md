# Optical Spiking Neural Networks via Rogue-Wave Statistics

This repository contains the source code, data, and simulation environment for the manuscript **"Optical Spiking Neural Networks via Rogue-Wave Statistics"** by Kesgin et al.

## 📂 Content
* **`optical_net_train.py`**: A PyTorch-based Optical Neural Network (ONN) for BreastMNIST dataset. It trains a phase layer to modulate light, aiming to generate rogue waves at specific spatial locations for image classification tasks.
* **`rogue_wave_analysis.py`**: Contains rogue wave simulations with data and control pattern using complex amplitude modulation.
* **`utils.py`**: Contains core physical functions, including the Angular Spectrum Method propagator and padding utilities. Angular Spectrum Method implementation is altered from https://github.com/computational-imaging/neural-holography.

## 🔗 DOI
https://doi.org/10.5281/zenodo.18109590

## 📜 License and Citation

This project is licensed under the **Creative Commons Attribution-NonCommercial (CC BY-NC)** license. The data and code are available for non-commercial research purposes only.

If you use this code or data in your research, please cite the following manuscript:

Kesgin, B. U., Durdu, G. Y., & Teğin, U. (2025). Optical Spiking Neural Networks via Rogue-Wave Statistics. arXiv preprint arXiv:2512.24983. doi: https://doi.org/10.48550/arXiv.2512.24983.

---
###  Requirements

To run the optical simulations, the following Python packages are required:

```bash
pip install numpy torch torchvision matplotlib scipy scienceplots tqdm scikit-learn medmnist scikit-image 
