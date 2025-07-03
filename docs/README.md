MSCPT （IEEE TMI 2025）
===========
MSCPT: Few-shot Whole Slide Image Classification with Multi-scale and Context-focused Prompt Tuning.

[Arxiv](https://arxiv.org/abs/2408.11505)
<!-- [ArXiv](https://arxiv.org/abs/2004.09666) | [Journal Link](https://www.nature.com/articles/s41551-020-00682-w) | [Interactive Demo](http://clam.mahmoodlab.org) | [Cite](#reference)  -->

<img src="logo.png" width="400px" align="right" />

**Abstract:** Multiple instance learning (MIL) has become a standard paradigm for weakly supervised classification of whole slide images (WSI). However, this paradigm relies on the use of a large number of labelled WSIs for training. The lack of training data and the presence of rare diseases present significant challenges for these methods. Prompt tuning combined with the pre-trained Vision-Language models (VLMs) is an effective solution to the Few-shot Weakly Supervised WSI classification (FSWC) tasks. Nevertheless, applying prompt tuning methods designed for natural images to WSIs presents three significant challenges: 1) These methods fail to fully leverage the prior knowledge from the VLM's text modality; 2) They overlook the essential multi-scale and contextual information in WSIs, leading to suboptimal results; and 3) They lack exploration of instance aggregation methods. To address these problems, we propose a Multi-Scale and Context-focused Prompt Tuning (**MSCPT**) method for FSWC tasks. Specifically, MSCPT employs the frozen large language model to generate pathological visual language prior knowledge at multi-scale, guiding hierarchical prompt tuning. Additionally, we design a graph prompt tuning module to learn essential contextual information within WSI, and finally, a non-parametric cross-guided instance aggregation module has been introduced to get the WSI-level features. Extensive experiments, visualizations, and interpretability analyses were conducted on five datasets and three downstream tasks using three VLMs, demonstrating the strong performance of our MSCPT.


---

<img src="overall.png" scaledwidth="100%" align="center" />

## Installation
- You can follow the [CLAM installation guide](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md) to install the dependency pack.
- If you want to optimize the prompt tuning for CONCH, please install [CONCH](https://github.com/mahmoodlab/CONCH) in the current folder.

## Updates

- **04/04/2025**: The support for [CONCH](https://github.com/mahmoodlab/CONCH) has been updated.
- **04/04/2025**: The support for UBC-OCEAN, PANDA and TCGA-BRCA-Recurrence has been updated.

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
In this paper, visual prompt tuning is not applied to the patches at 20x magnification. Instead, pre-extracted features from these patches are used as the visual input under 20x magnification. The feature extraction process follows the [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md) framework. Notably, we utilize [PLIP](https://github.com/PathologyFoundation/plip), [CLIP(ViT-B/16)](https://github.com/openai/CLIP) and [CONCH](https://github.com/mahmoodlab/CONCH) as feature extractors, requiring modifications to the original code. Upon completion of the feature extraction, you will obtain files with the following data structures:

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
## Acknowledgements
This project was funded by the National Natural Science Foundation of China 82090052.
If you find our work useful in your research or if you use parts of this code please cite our paper:
```bibtext
@article{han2024mscpt,
  title={MSCPT: Few-shot Whole Slide Image Classification with Multi-scale and Context-focused Prompt Tuning},
  author={Han, Minghao and Qu, Linhao and Yang, Dingkang and Zhang, Xukun and Wang, Xiaoying and Zhang, Lihua},
  journal={arXiv preprint arXiv:2408.11505},
  year={2024}
}
```
