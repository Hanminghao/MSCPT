import pandas as pd
import numpy as np
import os
import torch
import h5py
from torch.utils.data import  Dataset
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os
import openslide
import glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_pretrained_tokenizer(model_name):
    if model_name == 'plip':
        model_name = 'vinid/plip'
        tokenizer = AutoTokenizer.from_pretrained("path_to_plip_weight", use_fast=True)
    elif model_name == 'clip':
        model_name = 'openai/clip-vit-base-patch16'
        tokenizer = AutoTokenizer.from_pretrained('path_to_clip_weight', use_fast=True, TOKENIZERS_PARALLELISM=True)
    else:
        raise NotImplementedError
    return tokenizer

# load visual encoder weights and transforms
def load_ctranspath_clip(model_name, img_size = 224, return_trsforms = True):

    if model_name == 'plip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained("path_to_plip_weight")
        if return_trsforms:
            trsforms = get_transforms(img_size = img_size)
            return model, trsforms
    elif model_name == 'clip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('path_to_clip_weight')
        if return_trsforms:
            trsforms = get_transforms(img_size = img_size)
            return model, trsforms
    return model

def get_transforms(img_size=224, 
                            mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.225)):
    trnsfrms = transforms.Compose(
                    [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ]
                )
    return trnsfrms

def file_exists(df, root, ext = '.h5'):
    file_id = df['slide_id']
    df['has_h5'] = os.path.isfile(os.path.join(root, file_id + ext))
    return df

def read_assets_from_h5(h5_path):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs

def compute_patch_level(level0_mag, target_mag = 20, patch_size = 256):
    custom_downsample = int(level0_mag / target_mag)
    if custom_downsample == 1:
        target_level = 0
        target_patch_size = patch_size
    else:
        target_level = 0
        target_patch_size = int(patch_size * custom_downsample)
    return target_level, target_patch_size

def compute_patch_args(df, wsi_source, wsi_ext = '.svs', target_mag = 20, patch_size = 256):
    wsi_path = os.path.join(wsi_source, df['project_id'].split('-')[-1], df['slide_id'] + wsi_ext)
    # wsi = openslide.open_slide(wsi_path)
    df['patch_level'], df['patch_size'] = compute_patch_level(df['level0_mag'], target_mag, patch_size)
    return df

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, pt_features):
        self.coords = coords
        self.pt_features = pt_features

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        feature = self.pt_features[idx]
        return {'feature': feature, 'coords': coord}
    
@torch.no_grad()
def extract_features(df, model_name, model, wsi_ext = '.svs', pt_path='', device = 'cuda:0'):
    model.to(device)
    model.eval()
    slide_id = df['slide_id']
    wsi_path = os.path.join(wsi_source, df['project_id'].split('-')[-1], slide_id + wsi_ext)
    wsi = openslide.open_slide(wsi_path)
    pt_file_path = os.path.join(pt_path, df['project_id'].split('-')[-1], model_name+'_5', slide_id + '.pt')
    pt_file_path = pt_file_path.replace('clip', 'ViT-B-16')
    features = torch.load(pt_file_path).to(device)
    patch_level = df['patch_level']
    patch_size = df['patch_size']
    h5_path = os.path.join(h5_source, slide_id + '.h5')
    assets, _ = read_assets_from_h5(h5_path)
    return_coords = assets['coords']
    print(f'slide_id: {slide_id}, n_patches: {len(return_coords)}')

    with torch.no_grad():
        features = model.visual_projection(features)
        features = F.normalize(features, dim=-1) 
    return features, return_coords, wsi, patch_level, patch_size
            
import argparse
parser = argparse.ArgumentParser(description='Extract features using patch coordinates')
parser.add_argument('--csv_path', type=str, default='./tcga_rcc.csv', help='path to slide csv')
parser.add_argument('--h5_source', type=str,default = 'path_to_h5file', help='path to dir containing patch h5s')
parser.add_argument('--wsi_source', type=str,default = 'path_to_raw_wsi_file', help='path to dir containing wsis')
parser.add_argument('--pt_path', type=str, default='FEATURES_DIRECTORY', help='path to features')
parser.add_argument('--save_dir', type=str, default='path_to_save_dir', help='path to save extracted features')
parser.add_argument('--wsi_ext', type=str, default='.svs', help='extension for wsi')
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda:n')
parser.add_argument('--model_name', type=str, default='clip')
parser.add_argument('--gpt_data', type=str, default='./train_data/patch_selection/')
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--split', type=str, default='KIRC')
parser.add_argument('--dataset_name', type=str, default='RCC', choices=['Lung', 'RCC', 'BRCA'])
args = parser.parse_args()

def tokenize(tokenizer, texts):
    tokens = tokenizer.batch_encode_plus(texts, 
                                        max_length = 64,
                                        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                        return_token_type_ids=False,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

label_dicts = {
    'RCC': {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2},
    'Lung': {'LUAD': 0, 'LUSC': 1},
    'BRCA': {'IDC': 0, 'ILC': 1},
}

project_dic = {
    'RCC': ['KIRC', 'KIRP', 'KICH'],
    'Lung': ['LUAD', 'LUSC'],
    'BRCA': ['BRCA'],
}

if __name__ == '__main__':
    args.wsi_ext = '.svs'
    h5_source = args.h5_source
    wsi_source = args.wsi_source
    device = args.device
    prompt_file = os.path.join(args.gpt_data,f'{args.dataset_name.upper()}_select_pic.json')
    with open(prompt_file, 'r') as pf: 
        prompts = json.load(pf)

    model, trsforms = load_ctranspath_clip(model_name=args.model_name,
                                img_size = 224, 
                                return_trsforms = True)
    model.to(args.device)
    # Load tokenizer
    tokenizer = load_pretrained_tokenizer(args.model_name)
    all_weights = []
    for prompt_idx in range(len(prompts)):
        prompt = prompts[str(prompt_idx)]
        classnames = prompt['classnames']
        templates = prompt['templates']
        idx_to_class = {v:k for k,v in label_dicts[args.dataset_name].items()}
        n_classes = len(idx_to_class)
        classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames_text:
                texts = [template.replace('CLASSNAME', classname) for template in templates]

                texts, attention_mask = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
                texts = torch.from_numpy(np.array(texts)).to(device)
                attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
                class_embeddings = model.get_text_features(texts, attention_mask=attention_mask)
                
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        text_feats = torch.stack(zeroshot_weights, dim=0).to(device)
        all_weights.append(text_feats)
            
    text_feats = torch.stack(all_weights, dim=0).mean(dim=0)
    text_feats = F.normalize(text_feats, dim=-1)

    if not args.split:
        split_list = project_dic[args.dataset_name]
    else:
        split_list = [args.split]
    args.save_dir = os.path.join(args.save_dir, args.model_name, args.dataset_name)
    for args.split in split_list:
        df = pd.read_csv(args.csv_path)
        if args.split == 'KIRC':
            df = df[df['project_id']=='TCGA-KIRC']
        elif args.split == 'KIRP':
            df = df[df['project_id']=='TCGA-KIRP']
        elif args.split == 'KICH':
            df = df[df['project_id']=='TCGA-KICH']
        elif args.split == 'LUAD':
            df = df[df['project_id']=='TCGA-LUAD']
        elif args.split == 'LUSC':
            df = df[df['project_id']=='TCGA-LUSC']
        assert 'level0_mag' in df.columns, 'level0_mag column missing'
        h5_source = os.path.join(args.h5_source, args.split + '_5/patches')
        df = df.apply(lambda x: file_exists(x, h5_source), axis=1)
        df['has_h5'].value_counts()
        df = df[df['has_h5']]
        df = df.reset_index(drop=True)
        assert df['has_h5'].sum() == len(df['has_h5'])
        df['pred'] = np.nan 
        df = df.apply(lambda x: compute_patch_args(x, wsi_source, wsi_ext=args.wsi_ext, target_mag = 5, patch_size = 256), axis=1)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for idx in tqdm(range(len(df))):
            slide_id = df.iloc[idx]['slide_id']
            save_path = os.path.join(args.save_dir, slide_id)
            if os.path.exists(save_path):
                continue
            else:
                os.makedirs(save_path, exist_ok=True)
            category = label_dicts[args.dataset_name][df.iloc[idx]['OncoTreeCode']]
            img_feats, coords, wsi, patch_level, patch_size = extract_features( df.iloc[idx], model_name=args.model_name,
                                                                                wsi_ext=args.wsi_ext, 
                                                                                pt_path=args.pt_path,
                                                                                model=model, 
                                                                                device=device)
            logits = text_feats @ img_feats.T
            if args.top_k > img_feats.shape[0]:
                topk_values, topk_indices = torch.topk(logits, img_feats.shape[0], dim=1)
            else:
                topk_values, topk_indices = torch.topk(logits, args.top_k, dim=1)
            pred = topk_values.sum(dim=1).argmax().cpu().item()
            select_id = topk_indices.flatten().cpu().numpy()
            coord = coords[select_id]
            df.loc[idx, 'pred'] = 1 if pred == category else 0  # 进行赋值
            for idx, (x,y) in enumerate(coord):
                big_img = wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                big_img = big_img.resize((224,224))
                big_img.save(os.path.join(save_path, f"{idx}_{x}_{y}.png"))
