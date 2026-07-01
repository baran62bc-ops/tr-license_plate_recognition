import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import glob
from torchvision.transforms import v2
from tqdm.auto import tqdm
from torch.amp import autocast,GradScaler
import pandas as pd


dataset_path = "C:\\Users\\User\\OneDrive\\Masaüstü\\OCR\\oxford\\mnt\\ramdisk\\max\\90kDICT32px"
#dataset_path = "C:\\Users\\User\\Python\\ML\\Deep Learning\\Extracted Turkish License Plates"
chars = ['<blank>'] + list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
encode = lambda x : [char2idx[ch] for ch in x]
decode = lambda label : ''.join([idx2char[idx] for idx in label])

vocab_size = len(chars)
d_model = 256
max_sequence_length = 64
num_heads = 4
n_layers = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
# Recursively finds ALL .jpg files in any subdirectory
# NO NEED SINCE WE HAVE DS.PTH
#all_paths_og = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # (B, 64,  64, 128)

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # (B, 128, 32,  64)

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                        # (B, 256, 16,  64)

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Block 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                        # (B, 512,  8,  64)

            # Block 7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 64))              # (B, 512,  1,  64)
        )

    def forward(self, x):
        x = self.features(x)         # (B, 128, 1, W')
        x = x.squeeze(2)             # (B, 128, W')
        x = x.permute(0, 2, 1)       # (B, W', 128) = (B, T, C)
        return x

class AttentionHead(nn.Module): # single attention head
    def __init__(self,n_emb,head_size):
        super().__init__()
        self.query = nn.Linear(n_emb,head_size,bias=False)
        self.key = nn.Linear(n_emb,head_size,bias=False)
        self.value = nn.Linear(n_emb,head_size,bias=False)
        self.head_size= head_size
    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q_k = q @ k.transpose(-2,-1) * self.head_size**-0.5
        q_k = F.softmax(q_k,dim=-1)
        # No need to mask since we want the future information to affect the current state
        out = q_k @ v
        return out

class MultiHeadAttention(nn.Module): # one layer of transformer block
    def __init__(self,n_emb,num_heads):
        super().__init__()
        self.head_size = n_emb // num_heads # we have to make sure all are integers.
        self.blocks = nn.ModuleList([AttentionHead(n_emb,self.head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(n_emb,n_emb,bias=False)

    def forward(self,x):
        x = torch.cat([block(x) for block in self.blocks],axis=-1)
        x = self.linear(x)
        return x

class FeedForward(nn.Module): # one layer of transformer block
    def __init__(self,n_emb):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_emb,4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb,n_emb),
            nn.Dropout(0.1)
        )
    def forward(self,x):
        return self.ffwd(x)

class TransformerBlock(nn.Module): # one layer of transformer block
    def __init__(self,n_emb,num_heads):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_emb)
        self.layernorm2 = nn.LayerNorm(n_emb)
        self.mha = MultiHeadAttention(n_emb,num_heads)
        self.ffwd = FeedForward(n_emb)

    def forward(self,x):
        x = self.layernorm1(x + self.mha(x))
        x = self.layernorm2(x + self.ffwd(x))
        return x

class EncoderTransformerModel(nn.Module): # we will handle the ctcloss here.
    def __init__(self,n_emb,time_step,num_heads,head_size,n_layers,cnn_channels=128):
        super().__init__()
        self.n_emb = n_emb
        self.positional_encoding = nn.Embedding(max_sequence_length,n_emb)
        self.blocks = nn.Sequential(*[TransformerBlock(n_emb,num_heads)for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb,vocab_size) # z-a,Z-A,0-9,space, <unk> = 64
        self.input_proj = nn.Linear(512, n_emb, bias=False)
    def forward(self,idx,target=None,target_lengths = None):
        B,T,C = idx.shape # X -> CNN > [32, 32, 128] >  
        x = self.input_proj(idx)
        x = x + self.positional_encoding(torch.arange(T,device=idx.device))
        x = self.blocks(x)
        x = self.layernorm(x)
        logits = self.linear(x)
        # here comes the ctcloss
        loss = None
        if target is not None and target_lengths is not None:
            # CTC loss expects (T, B, vocab_size), log_probs, and input/target lengths
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=idx.device)
            #target_lengths = torch.sum(target != 0, dim=-1)  # non-blank chars per sample
            loss = F.ctc_loss(log_probs, target, input_lengths, target_lengths, blank=0)

        return logits, loss

class OCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.transformer = EncoderTransformerModel(
            n_emb=d_model,
            time_step=max_sequence_length,
            num_heads=num_heads,
            head_size=d_model // num_heads,
            n_layers=n_layers
        )

    def forward(self, x,target=None,target_lengths=None):
        x = self.cnn(x)                        # (B, T, 128)
        logits, loss = self.transformer(x, target,target_lengths)
        return logits, loss

class OCRDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L')
        label = path.split("\\")[-1].split("_")[1].split(".")[0]  # extract word from filename

        if self.transform:
            img = self.transform(img)

        encoded = torch.tensor(encode(label), dtype=torch.long)
        return img, encoded
