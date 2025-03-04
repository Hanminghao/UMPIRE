import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim
from ._utils import complete_masking
from ._visiumformer import CosineWarmupScheduler, Visiumformer
from transformers import ViTModel
import torch.nn.functional as F
import numpy as np
import timm  
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

CLS_TOKEN = 2

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Umpire(pl.LightningModule):
    def __init__(self, 
                 spot_config: dict,
                 visual_config: dict,
                 dim_output: int,
                 temperature: float,
                 extract_layers: List[int],
                 function_layers: str,
                 lr: float, 
                 warmup: int, 
                 max_epochs: int,
                 pool: int = 'mean',
                 without_context: bool = True,
                 margin: float = 0.5,
                 p: int = 2,
                 eps: float = 1e-6,
                 adapter: bool = False
                 ):
        """
        Args:
            backbone (pl.LightningModule): pretrained model
            baseline (bool): just for wandb logger to know it's baseline; baseline here means non-trained Transformer
            extract_layers (int): which hidden representations use as input for the linear layer
            function_layers (str): which function use to combine the hidden representations used
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token that gathers info of the sequence, mean just averages all tokens

        """
        super().__init__()
        self.spot_backbone = Visiumformer(dim_model=spot_config['dim_model'], 
                                            nheads=spot_config['nheads'], 
                                            dim_feedforward=spot_config['dim_feedforward'], 
                                            nlayers=spot_config['nlayers'],
                                            dropout=spot_config['dropout'],
                                            batch_first=spot_config['batch_first'], 
                                            n_tokens=spot_config['n_tokens'],
                                            context_length=spot_config['context_length'],
                                            autoregressive=spot_config['autoregressive'],
                                            pool=spot_config['pool'],
                                            learnable_pe=spot_config['learnable_pe'],
                                            masking_p=0.0)
        
        self.spot_backbone.hparams.masking_p = 0.0
        self.spot_projection = nn.Linear(self.spot_backbone.hparams.dim_model, dim_output)

        if spot_config['pretrained_path'] is not None:
            print("Loading pretrained spot model from", spot_config['pretrained_path'])
            checkpoint = torch.load(spot_config['pretrained_path'], map_location='cpu')
            self.spot_backbone.load_state_dict(checkpoint['state_dict'])

        if visual_config['model_name'] == 'phikon':
            print("Loading pretrained visal model")
            self.visual_backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            self.visual_projection = nn.Linear(self.visual_backbone.config.hidden_size, dim_output)
        elif visual_config['model_name'] == 'uni':
            self.visual_backbone = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            self.visual_projection = nn.Linear(self.visual_backbone.embed_dim, dim_output)
        elif visual_config['pretrained_path'] is not None and visual_config['model_name'] == 'conch':
            from conch.open_clip_custom import create_model_from_pretrained
            model =  create_model_from_pretrained("conch_ViT-B-16", 
                                 checkpoint_path=visual_config['pretrained_path'], 
                                 force_image_size=224,
                                 return_transform=False)
            self.visual_backbone  = model.visual
            self.visual_projection = nn.Linear(self.visual_backbone.proj_contrast.shape[0], dim_output)

        self.visual_backbone.train()
        self.visual_backbone_name = visual_config['model_name']
        
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.adapter = adapter
        if adapter:
            self.adapter_visual = Adapter(dim_output, 4)
            self.adapter_spot = Adapter(dim_output, 4)
        self.save_hyperparameters(ignore=['backbone'])
        
    def encode_gene(self, batch):
        # x -> size: batch x (context_length) x 1
        batch = complete_masking(batch, 0.0, self.spot_backbone.hparams.n_tokens+5)
        masked_indices = batch['masked_indices'].to(self.spot_backbone.device)
        attention_mask = batch['attention_mask'].to(self.spot_backbone.device)
        token_embedding = self.spot_backbone.embeddings(masked_indices)

        if self.spot_backbone.hparams.learnable_pe:
            pos_embedding = self.spot_backbone.positional_embedding(self.spot_backbone.pos.to(token_embedding.device))
            embeddings = self.spot_backbone.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.spot_backbone.positional_embedding(token_embedding)

        hidden_repr = []

        for i in range(len(self.spot_backbone.encoder.layers)):
            layer = self.spot_backbone.encoder.layers[i]
            embeddings = layer(embeddings, is_causal=self.spot_backbone.autoregressive, src_key_padding_mask=attention_mask) # bs x seq_len x dim
            if i in self.hparams.extract_layers:
                hidden_repr.append(embeddings)

        if self.hparams.function_layers == "mean":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.mean(combined_tensor, dim=-1)  # bs x seq_len x dim
        if self.hparams.function_layers == "sum":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.sum(combined_tensor, dim=-1)  # bs x seq_len x dim
        if self.hparams.function_layers == "concat":
            transformer_output = torch.cat(hidden_repr, dim=2)
                        

        if self.hparams.without_context:
            cls_prediction = transformer_output[:, 3:, :].mean(1)
        else:
            cls_prediction = transformer_output.mean(1)

        return cls_prediction
            
    def encode_visual(self, batch):
        # x -> size: batch x (context_length) x 1

        image = batch['images']
        
        if self.visual_backbone_name == 'phikon':
            outputs = self.visual_backbone(image)
            visual_features = outputs.last_hidden_state[:, 0, :]
        elif self.visual_backbone_name == 'uni':
            outputs = self.visual_backbone(image)
            visual_features = outputs
        elif self.visual_backbone_name == 'conch':
            visual_features = self.visual_backbone.forward_no_head(image, False)

        return visual_features
    
    def forward(self, batch):
        spot_features = self.encode_gene(batch)
        image_features = self.encode_visual(batch)
        
        spot_embeddings = self.spot_projection(spot_features)
        image_embeddings = self.visual_projection(image_features)
        if self.adapter:
            spot_embeddings_adapter = self.adapter_spot(spot_embeddings)
            image_embeddings_adapter = self.adapter_visual(image_embeddings)
            spot_embeddings = 0.8*spot_embeddings + 0.2*spot_embeddings_adapter
            image_embeddings = 0.8*image_embeddings + 0.2*image_embeddings_adapter

        return spot_embeddings, image_embeddings
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        # get the embeddings
        spot_embeddings, image_embeddings = self.forward(batch)
        # normalized features
        spot_embeddings = F.normalize(spot_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ spot_embeddings.t()
        logits_per_spot = logits_per_image.t()
        labels = torch.arange(logits_per_image.shape[0], device=self.device, dtype=torch.long)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_spot, labels)
        ) / 2

        self.log('train_loss', loss.mean(), sync_dist=True, prog_bar=True, reduce_fx='mean')

        
        return loss.mean()
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # get the embeddings
        spot_embeddings, image_embeddings = self.forward(batch)
        # normalized features
        spot_embeddings = F.normalize(spot_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ spot_embeddings.t()
        logits_per_spot = logits_per_image.t()
        labels = torch.arange(logits_per_image.shape[0], device=self.device, dtype=torch.long)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_spot, labels)
        ) / 2 
       
        self.log('val_loss', loss.mean(), sync_dist=True, prog_bar=True, reduce_fx='mean')
        
        return loss.mean()
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        
        data_key = 'tokenized_gene'

        if self.hparams.pool == 'cls': # Add cls token at the beginning of the set
            x = batch[data_key]
            cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device)*CLS_TOKEN # CLS token is index 2
            x = torch.cat((cls, x), dim=1) # add CLS
            batch[data_key] = x

        batch['tokenized_gene'] = batch['tokenized_gene'][:, :self.spot_backbone.hparams.context_length]
        
        return batch
    
    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.001)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_epochs=self.hparams.max_epochs)
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        
    def initialize_weights(self):

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0.0, std=0.02)
    
    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    





