from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from StereoData import StereoData
from torch.optim import AdamW # note the use of AdamW
from torch.utils.data import DataLoader
import model_utils as mu
import tqdm
import torch
import config
from pt_model import GPT2PromptTuningLM


EPOCHS = 4
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

args = config.Config()

# Initialize Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(['<antistereo>:', '<stereo>:', '<nonseq>:'])

# Handle Data
stereoData = StereoData("./stereoset.json", tokenizer)
stereoData =  DataLoader(stereoData, batch_size=BATCH_SIZE)

model = GPT2PromptTuningLM.from_pretrained(
    "gpt2",
    n_tokens=args.n_prompt_tokens,
    initialize_from_vocab=args.init_from_vocab
)

model.resize_token_embeddings(len(tokenizer))

model = model.to(DEVICE)
model.train()

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
        "weight_decay": args.weight_decay,
    }
]

optim = AdamW(optimizer_grouped_parameters, lr=1e-3)
#print("training...")
#mu.pt_train(stereoData, model, optim, EPOCHS, DEVICE)
model.eval()
TEST_PROMPT = "Many people live in Ethiopia."
output = mu.pt_infer(TEST_PROMPT, model, tokenizer, DEVICE, gen_code='n')
print(output)
