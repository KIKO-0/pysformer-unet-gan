# SmaAt-UNet
Code for the Paper "SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture" [Arxiv-link](https://arxiv.org/abs/2007.04417), [Elsevier-link](https://www.sciencedirect.com/science/article/pii/S0167865521000556?via%3Dihub)

![SmaAt-UNet](SmaAt-UNet.png)

The proposed SmaAt-UNet can be found in the model-folder under [SmaAt_UNet](models/SmaAt_UNet.py).

## Current workflow in this repo
This checkout has been adapted to a CIKM radar nowcasting workflow based on H5 data and Lightning checkpoints.

Main entrypoints:
- `build_cikm_h5.py`: convert raw CIKM text files into the H5 format used by training and inference.
- `train_precip_lightning.py`: first-stage precipitation training. The current default path trains `PhysFormerUNet`.
- `train_gan.py`: second-stage GAN fine-tuning on top of a first-stage checkpoint.
- `app.py`: Streamlit demo that loads a checkpoint and runs autoregressive inference on the H5 test split.

Expected local artifacts:
- Dataset H5: `data/CIKM/cikm_oversampled_v2.h5`
- First-stage checkpoints: `lightning/precip_regression/.../*.ckpt`
- Second-stage checkpoints: `lightning/precip_gan/.../*.ckpt`

These large data and checkpoint artifacts are intentionally ignored by git.

What this repository contains:
- Source code for H5 dataset conversion, first-stage training, second-stage GAN fine-tuning, and the Streamlit demo.
- Documentation and paper draft files under `docs/`.

What this repository does not contain:
- The full CIKM dataset text dumps.
- The generated H5 dataset.
- Trained Lightning checkpoints.
- Training logs, TensorBoard runs, or local cache files.

This means: cloning the repository and installing dependencies is enough to inspect the code, but not enough to run training or inference end to end. Before the scripts or app can run, you must place the dataset and checkpoints in the expected local paths.

## Required local files before running
Minimum files needed for each task:

- To build the H5 dataset:
  - `data/CIKM/train.txt`
  - `data/CIKM/testA.txt`

- To train the first stage:
  - `data/CIKM/cikm_oversampled_v2.h5`

- To train the second stage:
  - `data/CIKM/cikm_oversampled_v2.h5`
  - At least one first-stage checkpoint under `lightning/precip_regression/.../*.ckpt`

- To run the Streamlit demo:
  - `data/CIKM/cikm_oversampled_v2.h5`
  - At least one checkpoint under either:
    - `lightning/precip_regression/.../*.ckpt`
    - `lightning/precip_gan/.../*.ckpt`

Recommended directory layout:
```text
SmaAt-UNet/
├─ app.py
├─ build_cikm_h5.py
├─ train_precip_lightning.py
├─ train_gan.py
├─ data/
│  └─ CIKM/
│     ├─ train.txt
│     ├─ testA.txt
│     └─ cikm_oversampled_v2.h5
└─ lightning/
   ├─ precip_regression/
   │  └─ PhysFormerUNet/
   │     └─ *.ckpt
   └─ precip_gan/
      └─ *.ckpt
```

## Quick start
Install dependencies:
```shell
uv sync --frozen
```

Or use the generated requirements file:
```shell
python -m pip install -r requirements.txt
```

Build the CIKM H5 dataset:
```shell
python build_cikm_h5.py --train_txt data/CIKM/train.txt --test_txt data/CIKM/testA.txt --out_h5 data/CIKM/cikm_oversampled_v2.h5
```

Train the first stage:
```shell
python train_precip_lightning.py --dataset_folder data/CIKM/cikm_oversampled_v2.h5
```

Train the second stage:
```shell
python train_gan.py --dataset_folder data/CIKM/cikm_oversampled_v2.h5
```

Run the Streamlit demo:
```shell
python -m streamlit run app.py
```

The app scans both `lightning/precip_regression` and `lightning/precip_gan` for available checkpoints and runs inference on the H5 test set.

## Pre-run checklist
Before starting the app, confirm:

1. `data/CIKM/cikm_oversampled_v2.h5` exists.
2. `lightning/precip_regression` or `lightning/precip_gan` contains at least one `.ckpt`.
3. Your Python environment has `torch`, `lightning`, `h5py`, `streamlit`, `matplotlib`, and `numpy`.

Useful checks:
```shell
python -c "import os; print(os.path.exists('data/CIKM/cikm_oversampled_v2.h5'))"
python -c "import glob; print(glob.glob('lightning/precip_regression/**/*.ckpt', recursive=True))"
python -c "import glob; print(glob.glob('lightning/precip_gan/**/*.ckpt', recursive=True))"
```

## Common failure cases
`Dataset file not found: data/CIKM/cikm_oversampled_v2.h5`

The H5 dataset has not been generated or is not in the expected path. Run `build_cikm_h5.py` first or move the file into `data/CIKM/`.

`No checkpoint files were found under lightning/precip_regression or lightning/precip_gan`

The app found no trained model. Train the first stage first, or copy an existing checkpoint into one of those folders.

The app starts but shows no useful prediction choices

This usually means the H5 file exists, but the checkpoint folders are empty.

Training falls back to CPU

This repo currently includes a compatibility fallback for machines where the installed PyTorch build does not support the local GPU architecture. Training still runs, but slower.

**>>>IMPORTANT<<<**

The original Code from the paper can be found in this branch: https://github.com/HansBambel/SmaAt-UNet/tree/snapshot-paper

The current master branch has since upgraded packages and was refactored. Since the exact package-versions differ the experiments may not be 100% reproducible.

If you have problems running the code, feel free to open an issue here on Github.

## Installing dependencies
This project is using [uv](https://docs.astral.sh/uv/) as dependency management. Therefore, installing the required dependencies is as easy as this:
```shell
uv sync --frozen
```

If you have Nvidia GPU(s) that support CUDA, and running the command above have not installed the torch with CUDA support, you can install the relevant dependencies with the following command:
```shell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

In any case a [`requirements.txt`](requirements.txt) file are generated from `uv.lock`, which implies that `uv` must be installed in order to perform the export:
```shell
uv export --format=requirements-txt --no-dev --no-hashes --no-sources --output-file=requirements.txt
```

A `requirements-dev.txt` file is also generated for development dependencies:
```shell
uv export --format=requirements-txt --group dev --no-hashes --no-sources --output-file=requirements-dev.txt
```

A `requirements-full.txt` file is also generated for all dependencies (including development and additional dependencies):
```shell
uv export --format=requirements-txt --all-groups --no-hashes --no-sources --output-file=requirements-full.txt
```

Alternatively, users can use the existing [`requirements.txt`](requirements.txt), [`requirements-dev.txt`](requirements-dev.txt) and [`requirements-full.txt`](requirements-full.txt) files.

Basically, only the following requirements are needed for the training:
```
tqdm
torch
lightning
matplotlib
tensorboard
torchsummary
h5py
numpy
```

Additional requirements are needed for the [testing](###Testing):
```
ipykernel
```

---
For the paper we used the [Lightning](https://github.com/Lightning-AI/lightning) -module (PL) which simplifies the training process and allows easy additions of loggers and checkpoint creations.
In order to use PL we created the model [UNetDSAttention](models/unet_precip_regression_lightning.py) whose parent inherits from the pl.LightningModule. This model is the same as the pure PyTorch SmaAt-UNet implementation with the added PL functions.

### Training
An example [training script](train_SmaAtUNet.py) is given for a classification task (PascalVOC).

For training on the precipitation task we used the [train_precip_lightning.py](train_precip_lightning.py) file.
The training will place a checkpoint file for every model in the `default_save_path` `lightning/precip_regression`.
After finishing training place the best models (probably the ones with the lowest validation loss) that you want to compare in another folder in `checkpoints/comparison`.

### Testing
The script [calc_metrics_test_set.py](calc_metrics_test_set.py) will use all the models placed in `checkpoints/comparison` to calculate the MSEs and other metrics such as Precision, Recall, Accuracy, F1, CSI, FAR, HSS.
The results will get saved in a json in the same folder as the models.

The metrics of the persistence model (for now) are only calculated using the script [test_precip_lightning.py](test_precip_lightning.py). This script will also use all models in that folder and calculate the test-losses for the models in addition to the persistence model.
It will get handled by [this issue](https://github.com/HansBambel/SmaAt-UNet/issues/28).

### Plots
Example code for creating similar plots as in the paper can be found in [plot_examples.ipynb](plot_examples.ipynb).

You might need to make your kernel discoverable so that Jupyter can detect it. To do so, run the following command in your terminal:

```shell
uv run ipython kernel install --user --env VIRTUAL_ENV=$(pwd)/.venv --name=smaat_unet --diplay-name="SmaAt-UNet Kernel"
```

### Precipitation dataset
The dataset consists of precipitation maps in 5-minute intervals from 2016-2019 resulting in about 420,000 images.

The dataset is based on radar precipitation maps from the [The Royal Netherlands Meteorological Institute (KNMI)](https://www.knmi.nl/over-het-knmi/about).
The original images were cropped as can be seen in the example below:
![Precip cutout](Precipitation%20map%20Cutout.png)

If you are interested in the dataset that we used please write an e-mail to: k.trebing@alumni.maastrichtuniversity.nl and s.mehrkanoon@uu.nl

The 50% dataset has 4GB in size and the 20% dataset has 16.5GB in size. Use the [create_dataset.py](create_datasets.py) to create the two datasets used from the original dataset.

The dataset is already normalized using a [Min-Max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)).
In order to revert this you need to multiply the images by 47.83; this results in the images showing the mm/5min.

### Citation
```
@article{TREBING2021,
title = {SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture},
journal = {Pattern Recognition Letters},
year = {2021},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2021.01.036},
url = {https://www.sciencedirect.com/science/article/pii/S0167865521000556},
author = {Kevin Trebing and Tomasz Staǹczyk and Siamak Mehrkanoon},
keywords = {Domain adaptation, neural networks, kernel methods, coupling regularization},
abstract = {Weather forecasting is dominated by numerical weather prediction that tries to model accurately the physical properties of the atmosphere. A downside of numerical weather prediction is that it is lacking the ability for short-term forecasts using the latest available information. By using a data-driven neural network approach we show that it is possible to produce an accurate precipitation nowcast. To this end, we propose SmaAt-UNet, an efficient convolutional neural networks-based on the well known UNet architecture equipped with attention modules and depthwise-separable convolutions. We evaluate our approaches on a real-life datasets using precipitation maps from the region of the Netherlands and binary images of cloud coverage of France. The experimental results show that in terms of prediction performance, the proposed model is comparable to other examined models while only using a quarter of the trainable parameters.}
}
```
