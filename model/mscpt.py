import torch
from torch import nn
import json
import os
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

def create_model(
    model_name: str,
    override_image_size = None,
    pretrain_path = None
    ):  
    def clean_state_dict_ctranspath(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'attn_mask' in k:
                continue
            new_state_dict[k.replace('module.', '')] = v
        return new_state_dict
    
    if model_name == 'plip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('path_to_plip_weight')
        return model
    elif model_name == 'clip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('path_to_clip_weight')
        return model


def load_pretrained_tokenizer(model_name):
    if model_name == 'plip':
        model_name = 'vinid/plip'
        # tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
        tokenizer = AutoTokenizer.from_pretrained('path_to_plip_weight', use_fast=True, TOKENIZERS_PARALLELISM=True)
    elif model_name == 'clip':
        model_name = 'openai/clip-vit-base-patch16'
        # tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
        tokenizer = AutoTokenizer.from_pretrained('path_to_clip_weight', use_fast=True, TOKENIZERS_PARALLELISM=True)
    
    return tokenizer

class GcnPromptLearner(nn.Module):
    def __init__(self, traing):
        super(GcnPromptLearner, self).__init__()
        self.conv1 = GCNConv(512, 512)
        self.conv2 = GCNConv(512, 512)
        self.training = traing

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class MYPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_tpro, n_high, tokenizer):
        super().__init__()
        self.n_tpro = n_tpro # prompt length
        self.n_high = n_high # number of high-level prompts
        self.ctx_dim = clip_model.text_model.final_layer_norm.weight.shape[0]
        self.layers = len(clip_model.text_model.encoder.layers)
        self.dtype = clip_model.text_model.final_layer_norm.weight.dtype
        self.embeddings = clip_model.text_model.embeddings                                
        # global prompt for text encoder (except for the first layer)
        self.p_uni = nn.ParameterList([nn.Parameter(torch.empty(self.n_tpro, self.ctx_dim).type(self.dtype))
                                                      for _ in range(self.layers - 1)]) #[11,2,512]
        for p in self.p_uni:
            nn.init.normal_(p, std=0.02)
            
        # projector for learning high-level prompt (a.k.a p_ins)
        self.p_ins_projector = nn.Linear(self.ctx_dim, self.ctx_dim)
        
        # global prompt for the first layer of the text encoder
        self.p_input = nn.Parameter(torch.empty(self.n_tpro+self.n_high, self.ctx_dim))  #7,512
        nn.init.normal_(self.p_input, std=0.02)
        
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames)
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def forward(self, feats, desc):
        p_uni = self.p_uni
        prompts = []
        prompt_prefix = " ".join(["X"] * (self.n_tpro+self.n_high))

        for name in self.classnames:
            # We leverage all structures from descriptions as a part of input respectively during evaluation.
            for id in range(len(desc[name]['big_mag'])):
                p = prompt_prefix + ' ' + desc[name]['big_mag'][id]
                prompts.append(p)
    
        tokens = self.tokenizer(prompts, 
                                max_length = 64,
                                add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                return_token_type_ids=False,
                                truncation = True,
                                padding = 'max_length',
                                return_attention_mask=True)
        
        tokenized_prompts, attention_mask = tokens['input_ids'], tokens['attention_mask']
        tokenized_prompts = torch.from_numpy(np.array(tokenized_prompts)).cuda()
        attention_mask = torch.from_numpy(np.array(attention_mask)).cuda()

        with torch.no_grad():
            embedding = self.embeddings(tokenized_prompts).type(self.dtype)

        causal_attention_mask = _create_4d_causal_attention_mask(tokenized_prompts.size(), embedding.dtype, device=embedding.device)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, embedding.dtype)
        
        
        p_input = self.p_input.unsqueeze(0).expand(len(prompts), -1, -1)
        prefix = embedding[:, :1]
        suffix = embedding[:, 1+self.n_tpro+self.n_high:]
        
        # the input of the prompted text encoder
        p_ori = torch.cat([prefix, p_input, suffix], dim=1) #prefix， global prompts 以及 low-level promtps [num_class*num_desc, max_token_length, dim]

        # generate corresponding high-level prompt (p_ins)
        p_ins = []
        feats = feats.permute(1, 0, 2, 3)
        (l, c, n, d) = feats.shape
        feats = feats.reshape(l, c*n, d)
        for idx in range(self.layers - 1):
            feat = feats[idx].float()
            feat = feat + self.p_ins_projector(feat) 
            p_ins.append(feat)
        p_ins = torch.stack(p_ins, dim=0)
        self.tokenized_prompts = tokenized_prompts

        return p_ori, p_ins, p_uni, causal_attention_mask, attention_mask    #[the first layer global prompts & low-level prompts] [low-level prompts] [other layers global prompts] [relatation attention]


class VisionPromptLearner(nn.Module):
    def __init__(self, clip_model, n_vpro):
        super().__init__()
        self.n_vpro = n_vpro
        

        self.pro_dim = clip_model.vision_model.post_layernorm.weight.shape[0]
        self.dtype = clip_model.vision_model.post_layernorm.weight.dtype
        self.layers = len(clip_model.vision_model.encoder.layers)
        self.embeddings = clip_model.vision_model.embeddings


        self.p_visual = nn.ParameterList([nn.Parameter(torch.empty(self.n_vpro, self.pro_dim).type(self.dtype))
                                          for _ in range(self.layers-1)])
        for p in self.p_visual:
            nn.init.normal_(p, std=0.02)
            
        # global prompt for the first layer of image encoder
        self.p_input = nn.Parameter(torch.empty(self.n_vpro, self.pro_dim))
        nn.init.normal_(self.p_input, std=0.02)


    def forward(self, x):
        x = x.type(self.dtype)
        x = self.embeddings(x)
        
        p_input = self.p_input.unsqueeze(0).expand(len(x), -1, -1)
        x = torch.cat([x, p_input], dim=1)

        return x, self.p_visual

class VisionEncoder(nn.Module):
    def __init__(self, clip_model, n_vpro):
        super().__init__()
        self.n_vpro = n_vpro
        self.pre_layrnorm = clip_model.vision_model.pre_layrnorm
        self.encoder = clip_model.vision_model.encoder.layers
        self.post_layernorm = clip_model.vision_model.post_layernorm
        self.proj = clip_model.visual_projection
        self.dtype = clip_model.vision_model.post_layernorm.weight.dtype
        self.layers = clip_model.vision_model.encoder.layers

        

    def forward(self, x, p_visual):
        hidden_states = self.pre_layrnorm(x).type(self.dtype)
        for layer_idx, encoder_layer in enumerate(self.layers):
            if layer_idx > 0:
                hidden_states[-self.n_vpro:] = p_visual[layer_idx-1].unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            layer_outputs = encoder_layer(hidden_states, attention_mask=None,
                                            causal_attention_mask=None, output_attentions=False)    
            hidden_states = layer_outputs[0]    #[N,50,768]
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        out_put = self.proj(pooled_output)
        return out_put

class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_tpro, n_high):
        super().__init__()
        self.n_tpro = n_tpro # prompt length
        self.n_high = n_high
        self.num_layers = clip_model.text_model.encoder.layers # clip_model.transformer.resblocks
        self.positional_embedding = clip_model.text_model.embeddings.position_embedding #clip_model.positional_embedding
        self.final_layer_norm =  clip_model.text_model.final_layer_norm #clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = self.final_layer_norm.weight.dtype

    def forward(self, x, p_ins, p_uni, tokenized_prompts, attn, causal_attention_mask):
        (l, k, d) = p_ins.shape
        p_ins = p_ins.reshape(l, k//self.n_high, self.n_high, d) # [11, 3, 5, 512]   [num_layers, num_class, num_high_desc, dim]
        p_ins = p_ins.unsqueeze(2).expand(-1, -1, x.shape[0]//(k//self.n_high), -1, -1) # [11, 3, 20, 5, 512] [num_layers, num_class, num_low_desc, num_high_desc, dim]
        (l, num_class, num_low_desc, num_high_desc, d) = p_ins.shape
        p_ins = p_ins.reshape(l, num_class*num_low_desc, num_high_desc, d) # [11, 60, 5, 512] [num_layers, num_class*num_low_desc, num_high_desc, dim]
        p_ins = p_ins.type(self.dtype)
  
        for layer_idx, layer in enumerate(self.num_layers):
            if layer_idx > 0:               
                prefix = x[:, :1]
                suffix = x[: ,1+self.n_tpro+self.n_high:]
                
                # global-level prompt
                ctx_g = p_uni[layer_idx - 1].unsqueeze(0).expand(prefix.shape[0], self.n_tpro, -1)
                
                # high-level prompt
                ctx_l = p_ins[layer_idx - 1]
                x = torch.cat([prefix, ctx_g, ctx_l, suffix], dim=1)
                
                x = layer(x, attn, causal_attention_mask)[0]
                
            elif layer_idx == 0:
                x = layer(x, attn, causal_attention_mask)[0]
            else:
                x = layer(x)

        x = self.final_layer_norm(x)
        x = x[
                torch.arange(x.shape[0], device=x.device),
                tokenized_prompts.to(dtype=torch.int, device=x.device).argmax(dim=-1),
            ]
        x = self.text_projection(x)

        return x

class TextEncoderZS(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.text_model.encoder.layers
        self.embeddings = clip_model.text_model.embeddings
        self.final_layer_norm = clip_model.text_model.final_layer_norm
        self.text_projection = clip_model.text_projection


    def forward(self, input_ids, attention_mask=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        feats = []
        for _, layer in enumerate(self.encoder):
            layer_outputs = layer(hidden_states, attention_mask, causal_attention_mask)
            hidden_states = layer_outputs[0]
            # save class embeddings from different layers
            feats.append( hidden_states[
                torch.arange(hidden_states.shape[0], device=hidden_states.device),
                input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1),
            ])

        hidden_states = self.final_layer_norm(hidden_states)
        pooled_output = hidden_states[
                torch.arange(hidden_states.shape[0], device=hidden_states.device),
                input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1),
            ]
        out_put = self.text_projection(pooled_output)
        txt_feats = torch.stack(feats)

        return out_put, txt_feats   #out_put应该是[N,512]

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, description_dir, dataset_name,
                 base_model, n_high, n_tpro, n_vpro, LLM):
        super().__init__()
        # freaze all parmeters
        for p in clip_model.parameters():
            p.requires_grad = False
        # Load description and structure from gpt
        f_json = os.path.join(description_dir+f'/description/{LLM}', dataset_name+'.json')
        print(f'loading json file from: {f_json}')
        with open(f_json, 'r') as f:
            text_prompts = json.load(f)
        
        # tokenize
        tokenizer = load_pretrained_tokenizer(base_model)
        self.prompt_learner = MYPromptLearner(classnames, clip_model, n_tpro, n_high, tokenizer)
        self.gcn_prompt_learner_big = GcnPromptLearner(self.training)
        self.gcn_prompt_learner_small = GcnPromptLearner(self.training)
        self.vision_prompt_learner = VisionPromptLearner(clip_model, n_vpro)
        self.image_encoder = VisionEncoder(clip_model, n_vpro)
        self.text_encoder = TextEncoder(clip_model, n_tpro, n_high)
        self.text_encoder_zs = TextEncoderZS(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.model = clip_model
        self.n_class = len(classnames)

        def tokenize(tokenizer, texts):
            tokens = tokenizer.batch_encode_plus(texts, 
                                                max_length = 64,
                                                add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                                return_token_type_ids=False,
                                                truncation = True,
                                                padding = 'max_length',
                                                return_attention_mask=True)
            return tokens['input_ids'], tokens['attention_mask']
        
        with torch.no_grad():
            # zs_feats: layer-wise class embeddings from frozen text encoder
            # zs_repres: final representations from frozen text encoder
            zs_feats, zs_repres = [], []
            for classname in classnames:
                texts = text_prompts[classname]
                class_texts = texts['small_mag']

                class_texts, attention_mask = tokenize(tokenizer, class_texts)
                class_texts = torch.from_numpy(np.array(class_texts))
                attention_mask = torch.from_numpy(np.array(attention_mask))

                class_embeddings, features = self.text_encoder_zs(class_texts, attention_mask)
                class_embeddings = F.normalize(class_embeddings, dim=-1)
                features = F.normalize(features, dim=-1)
                zs_feats.append(features)
                zs_repres.append(class_embeddings)
            self.text_features_zs = torch.stack(zs_repres, dim=0).cuda()
            self.text_features_ft = torch.stack(zs_feats, dim=0).cuda() # [3, 11, 5, 512] [num_class, num_layers, num_desc, dim]
            self.text_prompts = text_prompts
            self.clip_model_proj = clip_model.visual_projection

    def forward(self, big_image, small_embeddings, train=True):
        big_image = big_image.squeeze(0)
        small_embeddings = small_embeddings.squeeze(0)
        small_embeddings = self.clip_model_proj(small_embeddings)
        small_embeddings = F.normalize(small_embeddings, dim=1)
        logit_scale = self.logit_scale.exp()
        
        text_features_zs = self.text_features_zs
        text_features_zs = text_features_zs.reshape(-1, text_features_zs.shape[-1])
        image_features_zs = small_embeddings
        image_features_zs = image_features_zs.reshape(-1, image_features_zs.shape[-1])

        p_ori, p_ins, p_uni, causal_attention_mask, attention_mask = self.prompt_learner(self.text_features_ft, self.text_prompts)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(p_ori, p_ins, p_uni, tokenized_prompts, attention_mask, causal_attention_mask)
        text_features = F.normalize(text_features, dim=1)
        
        # Since we use multiple structures for producing representations of one category, 
        # we should take their mean value as the final representation.
        x, p_visual = self.vision_prompt_learner(big_image)
        image_features = self.image_encoder(x, p_visual)
        image_features = F.normalize(image_features, dim=1)

        # asymmetric loss   
        sim_big = image_features @ text_features_zs.t()
        sim_big_ = sim_big


        sim_small = image_features_zs @ text_features.t()
        sim_small_ = sim_small

        sim_big_ = sim_big_.reshape(sim_big_.shape[0], self.n_class, -1).permute(1,0,2)
        A_big = torch.bmm(sim_big_, sim_big_.transpose(1,2))
        A_big = torch.max(A_big, dim=0)[0]
        A_big = torch.softmax(A_big, dim=-1)
        num_nodes_big = A_big.shape[0]
        edge_index_big = torch.tensor([[i, j] for i in range(num_nodes_big) for j in range(i+1, num_nodes_big)], dtype=torch.long).t().contiguous()
        x_big = self.gcn_prompt_learner_big(image_features, edge_index_big.cuda())

        sim_small_ = sim_small_.reshape(sim_small_.shape[0], self.n_class, -1)
        sim_small_ = sim_small_.permute(1,2,0)
        x_small = image_features_zs
        A_small = torch.bmm(sim_small_, sim_small_.transpose(1,2))
        A_small = torch.max(A_small, dim=0)[0]
        A_small = torch.softmax(A_small, dim=-1)
        num_nodes_small = A_small.shape[0]
        edge_index_small = torch.tensor([[i, j] for i in range(num_nodes_small) for j in range(i+1, num_nodes_small)], dtype=torch.long).t().contiguous()
        x_small = self.gcn_prompt_learner_small(x_small, edge_index_small.cuda())

        logits_i = x_big @ text_features_zs.t()
        return_sim_big = logits_i.reshape(logits_i.shape[0], self.n_class, -1).cpu().detach().numpy()
        logits_i = logits_i.reshape(-1, self.n_class)
        logits_i = logit_scale * torch.topk(logits_i, 5, dim=0)[0].mean(0)
        text_features_i = text_features.reshape(self.n_class, -1, text_features.shape[-1])
        text_features_i = text_features_i.mean(1)
        logits_i_cross = x_big @ text_features_i.t()
        logits_i_cross = logit_scale * torch.topk(logits_i_cross, 5, dim=0)[0].mean(0)
        logits_i = logits_i + logits_i_cross
        
        logits_t = x_small @ text_features.t()
        return_sim_small = logits_t.reshape(logits_t.shape[0], self.n_class, -1).cpu().detach().numpy()
        logits_t = logits_t.reshape(-1, self.n_class)
        logits_t = logit_scale * torch.topk(logits_t, 100, dim=0)[0].mean(0)
        text_features_t = text_features_zs.reshape(self.n_class, -1, text_features_zs.shape[-1])
        text_features_t = text_features_t.mean(1)
        logits_t_cross = x_small @ text_features_t.t()
        logits_t_cross = logit_scale * torch.topk(logits_t_cross, 100, dim=0)[0].mean(0)
        logits_t = logits_t + logits_t_cross
        

        logits = (logits_i + logits_t)/2

        if train:
            return logits, logits_i, logits_t
        else:
            return logits, (return_sim_small, return_sim_big)
class Mscpt(nn.Module):
    """ If you want to use pretrained model, or simply the standard structure implemented
        by Pytorch official, please use this template. It enable you to easily control whether
        use or not the pretrained weights, and whether to freeze the internal layers or not,
        and the in/out channel numbers, resnet version. This is made for resnet, but you can
        also adapt it to other structures by changing the `torch.hub.load` content.
    """
    def __init__(self, base_model='plip', base_pretrain_path='', trainer_perc='fp16', dataset_name='RCC',
                 description_dir='', label_dicts={}, n_set=5, n_tpro=2, n_vpro=2, n_high=10, n_topk=5, LLM='GPT_4'):
        super().__init__()

        classnames = [name for name in label_dicts.keys()]
        clip_model = create_model(model_name=base_model, pretrain_path=base_pretrain_path)
        if trainer_perc in ['fp32', 'amp']:
            clip_model.float()
        
        print("Building custom CLIP")
        self.Custom_model = CustomCLIP(classnames, clip_model, description_dir, dataset_name,
                                        base_model, n_high, n_tpro, n_vpro, LLM)
        
        
        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.Custom_model.named_parameters():
            if "prompt_learner"  not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.Custom_model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.Custom_model.parameters() if p.requires_grad)}")


    def forward(self, data, train=True):
    # def forward(self, data):
        big_img = data[0]
        small_embeddings = data[1]
        if train:
            logits, logits_i, logits_t = self.Custom_model(big_img, small_embeddings, True)
            return logits.unsqueeze(0), logits_i.unsqueeze(0), logits_t.unsqueeze(0)
        else:
            logits, return_sim = self.Custom_model(big_img, small_embeddings, False)
            return logits.unsqueeze(0), return_sim