MSCPT 
===========
MSCPT: Few-shot Whole Slide Image Classification with Multi-scale and Context-focused Prompt Tuning.


<!-- [ArXiv](https://arxiv.org/abs/2004.09666) | [Journal Link](https://www.nature.com/articles/s41551-020-00682-w) | [Interactive Demo](http://clam.mahmoodlab.org) | [Cite](#reference)  -->

***TL;DR:** MSCPT is a data-efficient method for applying vision-language models (VLMs) to few-shot whole slide image (WSI) classification. It leverages frozen large language models (LLMs) to generate visual descriptions of pathology at multiple scales, enhancing the ability of VLMs to classify WSIs more effectively. MSCPT has been tested on three different WSI datasets using two distinct VLMs, demonstrating impressive performance.*


<img src="overall.png" scaledwidth="100%" align="center" />

## Installation
You can follow the [CLAM installation guide](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md) to install the dependency pack.

## WSI Segmentation and Patching 
We also used CLAM for WSI segmentation and patching. For more details, please refer to the [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md). If you have successfully completed this step, you will get the h5 file with all the patch location information and generate the following folder structure at the specified H5_FILE_DIRECTORY:

```bash
H5_FILE_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```

The **masks** folder contains the segmentation results (one image per slide).
The **patches** folder contains arrays of extracted tissue patches from each slide (one .h5 file per slide, where each entry corresponds to the coordinates of the top-left corner of a patch)
The **stitches** folder contains downsampled visualizations of stitched tissue patches (one image per slide) (Optional, not used for downstream tasks)
The auto-generated csv file **process_list_autogen.csv** contains a list of all slides processed, along with their segmentation/patching parameters used.

## Feature Extraction
In this paper, visual prompt tuning is not applied to the patches at 20x magnification. Instead, pre-extracted features from these patches are used as the visual input under 20x magnification. The feature extraction process follows the [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md) framework. Notably, we utilize [PLIP](https://github.com/PathologyFoundation/plip) and [CLIP(ViT-B/16)](https://github.com/openai/CLIP) as feature extractors, requiring modifications to the original code. Upon completion of the feature extraction, you will obtain files with the following data structures:

```bash
FEATURES_DIRECTORY/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```
## Patch Selection
Since the computational cost of visual prompt tuning for all patches is unacceptable, we use the zero-shot capability of VLM to extract some patches as inputs for visual prompt tuning at 5x magnification. We first build prompts for patch selection:
```bash
python generate_select_prompt.py --dataset_name DATASET_NAME
```
Then run `select_5X_pic.py` to select patches: 
```bash
CUDA_VISIBLE_DEVICES=0 python select_5X_pic.py --k 30 --h5_source H5_FILE_DIRECTORY/patches --wsi_source WSI_FILE_DIRECTORY --pt_path FEATURES_DIRECTORY --save_dir SELECTED_PATCHES_DIRECTORY --model_name MODEL_NAME --dataset_name DATASET_NAME
```
If you complete this step, then you will have the following data structure:
```bash
SELECTED_PATCHES_DIRECTORY/
	├── slide_1
    		├── 0.png
    		├── 1.png
    		└── ...
	├── slide_2
    		├── 0.png
    		├── 1.png
    		└── ...
	├── slide_3
    		├── 0.png
    		├── 1.png
    		└── ...
	└── ...
```
## Dataset Spliting
According to the description in the paper, we divided each dataset into training set (20%) and test set (80%), and randomly selected 16 training samples for each type in the training set. Run `numshots/get_train_val_split.py` to split the dataset:
```bash
python numshots/get_train_val_split.py --pt_path FEATURES_DIRECTORY --dataset_name DATASET_NAME --numshots 16
```
## Model Training
To facilitate the replication of our method, we wrote the `.sh` files for the three data sets to make it easy to run the program with one click:
```bash
bash scripts/mscpt/train_my_brca.sh 0
```
Note that you need to enter your SELECTED_PATCHES_DIRECTORY and FEATURES_DIRECTORY in the corresponding `.sh` file.
## Funding
This project was funded by the National Natural Science Foundation of China 82090052.
