import pandas as pd
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='RCC', choices=['Lung', 'RCC', 'BRCA'])
parser.add_argument('--numshots', type=int, default=16, choices=[0, 1, 2, 4, 8, 16])
parser.add_argument('--pt_path', type=str, default='FEATURES_DIRECTORY')
args = parser.parse_args()

csv_dir = f'./tcga_{args.dataset_name.lower()}.csv'
df = pd.read_csv(csv_dir)
categories = df['OncoTreeCode'].unique()
exist_pt_path = args.pt_path
exist_pt_slides = []
for file in os.listdir(exist_pt_path):
    exist_pt_slides.append(file.split('.pt')[0])
df = df[df.slide_id.isin(exist_pt_slides)]

for seed in range(1,6):
    val_df = pd.read_csv(f'./numshots/{args.dataset_name}/{args.dataset_name}_val_{args.numshots}_{seed}.csv')
    train_df = pd.read_csv(f'./numshots/{args.dataset_name}/{args.dataset_name}_train_{args.numshots}_{seed}.csv')
    train_df_ = df[~df.slide_id.isin(val_df.slide_id)]
    train_df_ = train_df_[~train_df_.slide_id.isin(train_df.slide_id)]
    val_df = val_df.reset_index(drop=True)
    for categorie in categories:
        samples = train_df_[train_df_['OncoTreeCode']==categorie].sample(n=args.numshots, random_state=seed)
        train_df = pd.concat([train_df, samples], axis=0)
    save_train_df = train_df.reset_index(drop=True)
    save_train_df.to_csv(f'./numshots/{args.dataset_name}/{args.dataset_name}_train_{args.numshots}_{seed}.csv', index=False)
    val_df.to_csv(f'./numshots/{args.dataset_name}/{args.dataset_name}_val_{args.numshots}_{seed}.csv', index=False)
