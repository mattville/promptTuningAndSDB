from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from StereoData import StereoData
from torch.optim import Adam
from torch.utils.data import DataLoader
import model_utils as mu
import tqdm
import torch

EPOCHS = 4
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(['<antistereo>:', '<stereo>:', '<nonseq>:'])

# Handle Data
stereoData = StereoData("./stereoset.json", tokenizer)
stereoData =  DataLoader(stereoData, batch_size=BATCH_SIZE)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

model = model.to(DEVICE)
"""print(tokenizer.decode(model.generate(**tokenizer("What are we waiting for?",
                                                  return_tensors="pt"))[0]))"""
model.train()
optim = Adam(model.parameters(), lr=1e-3)
print("training...")
mu.train(stereoData, model, optim, EPOCHS, DEVICE)