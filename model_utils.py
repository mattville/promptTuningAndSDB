import tqdm
import torch
from torch.nn import functional as F

# a: antistereotype, s: stereotype, n: nonsequitor, e: *inject empty*
LABELS = {'a': '<antistereo>:', 's': '<stereo>:', 'n': '<nonseq>:', 'e':''}

def train(data, model, optim, epochs, device):
    for i in tqdm.tqdm(range(epochs)):
        print("Beginning epoch " + str(i))
        for X, a in data:
            print("for loop initializes")
            X = X.to(device)
            a = a.to(device)
            print("devices are configured")
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            print("loss configured")
            loss.backward()
            optim.step()
            print("completed gd step")
        torch.save(model.state_dict(), "model_state.pt")


def infer(inp, model, tokenizer, device, gen_code='e'):
    inp = "<startofstring> " + inp + " " + LABELS[gen_code] + " "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a )
    output = tokenizer.decode(output[0])
    return output


def pt_train(data, model, optim, epochs, device):
    for i in tqdm.tqdm(range(epochs)):
        print("Beginning epoch " + str(i))
        for X, a in data:
            print("for loop initializes")
            X = X.to(device)
            a = a.to(device)
            print("devices are configured")
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            print("loss configured")
            loss.backward()
            optim.step() 
            print("completed gd step")
        #torch.save(model.state_dict(), "model_state.pt")

def pt_infer(inp, model, tokenizer, device, gen_code='e'):
    inp = "<startofstring> " + inp + " " + LABELS[gen_code] + " "
    inp = tokenizer(inp, return_tensors="pt")
    tokens = inp["input_ids"].to(device)
    with torch.no_grad():
      for i in range(10):
          outputs = model.forward(input_ids=tokens)
          #outputs = model(input_ids=tokens)
          next_token_logits = outputs[0][:, -1, :]
          probs = F.softmax(next_token_logits, dim = -1)
          next_tokens = torch.argmax(probs).unsqueeze(0)
          tokens = torch.cat([tokens.squeeze(), next_tokens], dim=0).unsqueeze(0)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)
