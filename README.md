# DM<sup>3</sup>T: Harmonizing Modalities via Diffusion for Multi-Object Tracking

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](MASK4REVIEW)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)](https://pytorch.org/)

</div>

## 🔄 Updates
- **[2025-11-19]**: New version, sample release for review.
- **[2025-08-05]**: Initial sample release for review.

## 🎯 Key Features

- **Cross-Modal Diffusion Fusion**: Novel cross-guided denoising where RGB and thermal features provide mutual guidance during the diffusion process
- **Diffusion Refiner**: Plug-and-play module to enhance and refine unified feature representations
- **Hierarchical Tracker**: Adaptive confidence estimation for improved tracking robustness
- **End-to-End Framework**: Unifies object detection, state estimation, and data association without complex post-processing
- **Real-time Performance**: Enables online tracking with temporal coherence

## 🏗️ Architecture Overview

![Architecture](assets/Framework.png)

DiffMM-Track consists of:
1. **Cross-Modal Diffusion Fusion**: Iterative cross-guided denoising between RGB and thermal modalities
2. **Diffusion Refiner**: Enhances unified feature representation through denoising processes
3. **Hierarchical Tracker**: Adaptively handles confidence estimation for robust tracking

## 📊 Performance

### VTMOT Benchmark Results

| Method | HOTA↑ | IDF1↑ | MOTA↑ | DetA↑ | MOTP↑ |
|--------|-------|-------|-------|-------|-------|
| FairMOT | 37.35 | 45.80 | 37.27 | 34.63 | 72.53 |
| CenterTrack | 39.05 | 44.42 | 30.59 | 38.10 | 72.87 |
| TransTrack | 38.00 | 43.57 | 36.16 | 35.71 | 73.82 |
| ByteTrack | 38.39 | 45.76 | 33.15 | 32.12 | 73.48 |
| OC-SORT | 31.48 | 38.09 | 28.95 | 25.24 | 73.15 |
| MixSort-OC | 39.09 | 45.80 | 31.33 | 33.11 | 73.63 |
| MixSort-Byte | 39.58 | 46.37 | 31.59 | 34.81 | 73.05 |
| PID-MOT | 35.62 | 42.43 | 33.33 | 33.25 | 71.79 |
| Hybrid-SORT | 39.49 | 46.31 | 31.07 | 34.62 | 72.84 |
| PFTrack | 41.07 | 47.25 | **43.09** | **41.63** | **73.95** |
| **DM<sup>3</sup>T** | **41.70** | **48.00** | 36.76 | 41.46 | 73.15 |


## 🚀 Getting Started

### Prerequisites

```bash
# Create conda environment
conda create -n dm3t python=3.8
conda activate dm3t

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Installation

1. **Clone the repository**
```bash
git clone [MASK4REVIEW]
cd dm3t
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Build DCNv2 (if needed)**
```bash
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```

4. **Install TrackEval**
```bash
cd trackeval
pip install -e .
```

### Dataset Preparation

#### VTMOT Dataset
1. Download the VTMOT dataset from [PFTrack](https://github.com/wqw123wqw/PFTrack)
2. Organize the dataset structure:
```
VTMOT/
├── images/
│   ├── train/
│   │   ├── $sequence_name$/
│   │   │   ├── gt/
│   │   │   └── infrared/
│   │   │   └── visible/
│   │   │   └── seqinfo.ini
│   │   └── ...
│   └── test/
│   │   ├── $sequence_name$/
│   │   │   ├── gt/
│   │   │   └── infrared/
│   │   │   └── visible/
│   │   │   └── seqinfo.ini
└── annotations/
    ├── train/
    └── test/
```

3. Convert annotations to COCO format:
```bash
python tools/convert_vtmot_to_coco.py
```

## 🏃‍♂️ Quick Start

> 🚧 **Coming Soon!** 🚧
> 
> We're actively working on providing:
> - ⚡ Ready-to-use training scripts
> - 🔄 One-click evaluation pipelines
> - 📊 Example configurations for different scenarios
>
> Stay tuned for updates!

## 📝 Model Zoo

> 🚧 **Coming Soon!** 🚧
>
> Our model collection is under preparation and will include:
> - 🔥 Pre-trained checkpoints for various architectures
> - ⚙️ Configuration files for different settings
> - 📈 Performance benchmarks and comparison charts
>
> Check back for regular updates!

## 🔧 Training Your Own Model

### Configuration

Edit the configuration in `src/lib/opts.py` or create your own experiment:

```bash
cd src
python main.py tracking --exp_id your_experiment --dataset vtmot --arch dla_34
```

### Training Parameters

- `--arch`: Backbone architecture (dla_34, resnet_101, vit, etc.)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--num_epochs`: Number of training epochs
- `--diffusion_refiner`: Using diffusion refiner module
- `--hierarchical_levels`: Levels in hierarchical tracker

## 📊 Evaluation

### Standard Evaluation

```bash
cd trackeval
python eval.py --BENCHMARK VTMOT --SPLIT_TO_EVAL test --TRACKERS_TO_EVAL DM3T
```

### Custom Evaluation

```bash
python tools/eval_mot.py --result_folder ../exp/tracking/results/ --dataset vtmot
```

## 📄 Citation

> 📑 **Coming Soon!**

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PFTrack](https://github.com/wqw123wqw/PFTrack) for dataset and model comparisons.
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) for evaluation metrics.


