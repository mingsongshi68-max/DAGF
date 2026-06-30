# DAGF: A Disentangled Adaptive Gating Fusion Network for Multimodal Sentiment Analysis

## Usage

### Prerequisites

- Python 3.9.13
- PyTorch 1.13.0
- CUDA 11.7

### Installation

- Create a conda environment. Please make sure you have installed conda before.

```bash
conda create -n DAGF python=3.9.13
```

- Activate the built DAGF environment.

```bash
conda activate DAGF
```

- Install PyTorch with CUDA

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

- Clone this repo.

```bash
git clone https://github.com/mingsongshi68-max/DAGF.git
```

- Install the necessary packages.

```bash
cd DAGF
pip install -r requirements.txt
```

## Datasets

Data files (containing processed MOSI, MOSEI datasets) can be downloaded from [here](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi). You can first build `./dataset` and then put the downloaded datasets into the directory and revise the path in.For more details, please follow the [official website ](https://github.com/ecfm/CMU-MultimodalSDK) of these datasets.

## Run the Codes

### Training

You can first set the training dataset name in `config.json` as "mosei" or "mosi", and then run:

```bash
python train.py
```

By default, the trained model will be saved in `./pt` directory. You can change this in `train.py`.

### Testing

You can first set the testing dataset name in `config.json` as "mosei" or "mosi", and then test the trained model:

```bash
python test.py
```

## Contact

- Mingsong Shi: mingsongshi68@163.com
- Junfeng Shen: shenjunfengwindy@163.com

## License

MIT License
