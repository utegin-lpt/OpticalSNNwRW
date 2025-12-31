# Optical Spiking Neural Networks via Rogue-Wave Statistics

This repository contains the source code, data, and simulation environment for the manuscript **"Optical Spiking Neural Networks via Rogue-Wave Statistics"** by Kesgin et al.

## Content
* **`optical_net_train.py`**: A PyTorch-based Optical Neural Network (ONN). It trains a diffractive phase layer to modulate light, aiming to generate rogue waves at specific spatial locations for image classification tasks (e.g., BreastMNIST).
* **`rogue_wave_analysis.py`**: Contains rogue wave simulations with data and control pattern using complex amplitude modulation.
* **`utils.py`**: Contains core physical functions, including the Angular Spectrum Method propagator and padding utilities. Angular Spectrum Method implementation is altered from https://github.com/computational-imaging/neural-holography.

## 🔗 DOI and Data
TBA

## 📜 License and Citation

This project is licensed under the **Creative Commons Attribution-NonCommercial (CC BY-NC)** license. The data and code are available for non-commercial research purposes only.

If you use this code or data in your research, please cite the following manuscript:

---
###  Requirements

To run the optical simulations, the following Python packages are required:

```bash
pip install numpy torch torchvision matplotlib scipy scienceplots tqdm
