# LETS Forecast: Learning Embedology for Timeseries Forecasting
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://abrarmajeedi.github.io/deep_edm/)
[![arXiv](https://img.shields.io/badge/arXiv-2312.04364-b31b1b.svg)](https://arxiv.org/abs/2506.06454)

Accepted at International Conference on Machine Learning (ICML) 2025!


## Abstract
Real-world time series are often governed by complex nonlinear dynamics. Understanding these underlying dynamics is crucial for precise future prediction. While deep learning has achieved major success in time series forecasting, many existing approaches do not explicitly model the dynamics. To bridge this gap, we introduce DeepEDM, a framework that integrates nonlinear dynamical systems modeling with deep neural networks. Inspired by empirical dynamic modeling (EDM) and rooted in Takens' theorem, DeepEDM presents a novel deep model that learns a latent space from time-delayed embeddings, and employs kernel regression to approximate the underlying dynamics, while leveraging efficient implementation of softmax attention and allowing for accurate prediction of future time steps. To evaluate our method, we conduct comprehensive experiments on synthetic data of nonlinear dynamical systems as well as real-world time series across domains. Our results show that DeepEDM is robust to input noise, and outperforms state-of-the-art methods in forecasting accuracy.

![](./pic/main_fig.png)

## Setup and Data

1. Set up the environment
```
conda create -n deepedm_env python=3.8

conda activate deepedm_env

pip install -r requirements.txt
```

2. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`.
3. Data can also be obtained from links in the awesome [Time-Series-Library](https://github.com/thuml/Time-Series-Library) repository.
  

## Usage

We provide the scripts to run all benchmarks under the folder `./scripts/`. For convenience, we have added helper scripts to run experiments quickly


## Main Results (i.e. Table 1 in the paper)

```
bash ./run_all.sh
```
or you can run individual dataset-forecast length experiments as:

```
# Args are: datasets, seed, pred_len_idx
# O refers to 48, 1 refers to 96 and so on
bash quick_test.sh ETTh1 2021 0
```
To modify params change the command-line args in the actual scripts within the `./scripts/long_term_forecast` directory.


## Short-term Forecasting (M4 i.e. Table 5)
```
bash quick_test_m4.sh
```

To modify params change the command-line args in the actual scripts within the `./scripts/short_term_forecast` directory.


## Citation
If you find our work useful, please cite us as:
```
@inproceedings{majeedi2025lets,
	title={{LETS} Forecast: Learning Embedology for Time Series Forecasting},
	author={Abrar Majeedi and Viswanatha Reddy Gajjala and Satya Sai Srinath Namburi GNVV and Nada Magdi Elkordi and Yin Li},
	booktitle={Forty-second International Conference on Machine Learning},
	year={2025},
	url={https://openreview.net/forum?id=LLk1qYQatJ}
}
```
## Acknowledgement

This repository relies heavily on the amazing [Time-Series-Library](https://github.com/thuml/Time-Series-Library) repository. We are thankful for their work. Please check them out!

## Contact
If you have any questions about our work or code, feel free to email:
```
 Abrar Majeedi (majeedi@wisc.edu)
```