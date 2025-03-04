import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from torch import optim
from ._utils import complete_masking
import numpy as np
import math
MASK_TOKEN = 0
CLS_TOKEN = 2


class Visiumformer(pl.LightningModule):
    
    def __init__(self, 
                 dim_model: int, 
                 nheads: int, 
                 dim_feedforward: int,
                 nlayers: int, 
                 dropout: float,
                 batch_first: bool, 
                 masking_p: float, 
                 n_tokens: int,
                 context_length: int,
                 lr: float = None, 
                 warmup: int = None, 
                 batch_size: int = None, 
                 max_epochs: int = None,
                 autoregressive: bool = None,
                 pool: str = None,
                 cls_classes: int = 164,
                 supervised_task: int = None,
                 learnable_pe: bool = True,
                 ):
        """
        Args:
            dim_model (int): Dimensionality of the model
            nheads (int): Number of attention heads
            dim_feedforward (int): Dimensionality of MLPs of attention blocks
            batch_first (int): batch first dimension
            masking_p (float): p value of Bernoulli for masking
            n_tokens (int): total number of tokens (WITHOUT auxiliar tokens)
            context_length (int): length of the context, who would have guessed... 
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            autoregressive (bool): if True, implements autoregressive training
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token at the beginning, mean just averages all tokens. If not supervised task during training, is ignored
            cls_classes (int): number of classes to classify
            supervised_task (str): None, 'classification' or 'regression'
            learnable_pe (bool): if True, positional embeddings are learnable embeddings, otherwise are derived from trigonometric functions
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nheads, dim_feedforward=dim_feedforward, batch_first=batch_first, dropout=dropout, layer_norm_eps=1e-12)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=nlayers, enable_nested_tensor=False)
        
        # As in HuggingFace
        self.classifier_head = nn.Linear(dim_model, n_tokens, bias=False)
        bias = nn.Parameter(torch.zeros(n_tokens)) # each token has its own bias
        self.classifier_head.bias = bias
            
        # As in HuggingFace
        self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
        self.cls_head = nn.Linear(dim_model, cls_classes)

        # Token embedding learnable weights
        self.embeddings = nn.Embedding(num_embeddings=n_tokens+5, embedding_dim=dim_model, padding_idx=1)
        
        if pool == 'cls':
            context_length += 1
            
        if not learnable_pe:
            self.positional_embedding = PositionalEncoding(d_model=dim_model, max_seq_len=context_length)
        else:
            self.positional_embedding = nn.Embedding(num_embeddings=context_length, embedding_dim=dim_model) 
            self.dropout = nn.Dropout(p=dropout)
            self.pos = torch.arange(0, context_length, dtype=torch.long)
        
        # MLM loss
        self.loss = nn.CrossEntropyLoss()
       
        if supervised_task is not None:
            self.cls_loss = nn.CrossEntropyLoss()
            
        self.autoregressive = autoregressive
        
        self.save_hyperparameters()

        self.gc_freq = 5
        
        self.batch_train_losses = []
        
        self.initialize_weights()

            
    def forward(self, x, attention_mask):
                
        # x -> size: batch x (context_length) x 1
        token_embedding = self.embeddings(x) # batch x (n_tokens) x dim_model

        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device)) # batch x (n_tokens) x dim_model        
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)
        
        transformer_output = self.encoder(embeddings, is_causal=self.autoregressive, src_key_padding_mask=attention_mask) # batch x (n_tokens) x dim_model

        # MLM prediction
        prediction = self.classifier_head(transformer_output)
            
        return {'mlm_prediction': prediction,
                'transformer_output': transformer_output}
    
    def training_step(self, batch, batch_idx, *args, **kwargs):

        with torch.no_grad():
            batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
            
        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['tokenized_gene']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        mlm_predictions = predictions['mlm_prediction']
        
        # we just evaluate on the masked tokens (mask = 0)
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)

        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens)
        real_indices = real_indices.view(-1)
        masked_indices = masked_indices.view(-1)

        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length
        
        if self.hparams.masking_p == 0.0: # this case is uniquely for the fine tuning case (check _fine_tune_model)
            loss = torch.tensor(0.0, device=mlm_predictions.device)
        else:
            loss = self.loss(mlm_predictions, real_indices) # MLM loss
                                          
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        with torch.no_grad():
            batch = complete_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)
            
        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['tokenized_gene']
        attention_mask = batch['attention_mask']

        predictions = self.forward(masked_indices, attention_mask)
        mlm_predictions = predictions['mlm_prediction']
        
        real_indices = torch.where(mask==MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(torch.int64)

        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens)
        real_indices = real_indices.view(-1)
        masked_indices = masked_indices.view(-1)

        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length
        
        if self.hparams.masking_p == 0.0: # this case is uniquely for the fine tuning case (check _fine_tune_model)
            loss = torch.tensor(0.0, device=mlm_predictions.device)
        else:
            loss = self.loss(mlm_predictions, real_indices) # MLM loss
        
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')
        
        return loss

    
        
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        
        data_key = 'tokenized_gene'

        if self.hparams.pool == 'cls': # Add cls token at the beginning of the set
            x = batch[data_key]
            cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device)*CLS_TOKEN # CLS token is index 2
            x = torch.cat((cls, x), dim=1) # add CLS
            batch[data_key] = x
        batch['tokenized_gene'] = batch['tokenized_gene'][:, :self.hparams.context_length]
        
        return batch
    
    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_epochs=self.hparams.max_epochs)
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
                
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_epochs):
        self.warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor




