# DeepDR: a deep learning library for drug response prediction

This repository is the official implementation.

## Install DeepDR

```
pip install DeepDR -i https://pypi.org/simple
```

## Note

Due to the PyPI upload file size limit, in the current version, DAE_ALL.pt is not integrated in the library. If you need, please download from https://huggingface.co/spaces/user15632/DeepDR/blob/main/DeepDR/DefaultData/DAE_ALL.pt.

## Content

additional: Contains the data and model you might need when using DeepDR.

deepdr: Contains the original files for all versions of the DeepDR library.

preprocess: Contains raw data, scripts and results for data preprocessing.

results: Contains scripts and results for benchmark experiments.

## Benchmark

The benchmark in the paper used the DeepDR v0.1, it can be downloaded from https://drive.google.com/file/d/1usL_HFmCfndN4hkHq97CR4Lj1JaxiMm_/view?usp=sharing (see deepdr/v0.1/download path.txt).

Run the following command in the result/datasets/ directory to split dataset:

```
python -m get_data
```

e.g. Run the following command in the result/tuning_CCLE_cell_out/ directory for model training and validation. 
Make sure the pkl files generated in the previous step are in the same path. 
You may need to download and unzip the file from https://drive.google.com/file/d/1iWPWElWHatE6ZR2vothSo6ZgeTpkgwz6/view?usp=drive_link (see additional/download path.txt), and place them under the same path:

```
python -m tuning_CCLE_cell_out_0_0
```

e.g. Run the following command in the same directory as the previous step to calculate the evaluation metrics:

```
python -m res_CCLE_cell_out
```
