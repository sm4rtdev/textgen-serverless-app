from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import torch.nn.functional as F
from random import choice
import requests
import gc


def get_pred(text, model, tok, p=0.7):
    input_ids = torch.tensor(tok.encode(text)).unsqueeze(0)
    _, logits, _ = model(input_ids, labels=input_ids)
    probs = F.softmax(logits[:, -1], dim=-1).squeeze()
    del logits, input_ids
    gc.collect()
    idxs = torch.argsort(probs, descending=True)
    res, cumsum = [], 0.
    for idx in idxs:
        res.append(idx)
        cumsum += probs[idx]
        if cumsum > p:
            pred_idx = idxs.new_tensor([choice(res)])
            break
    pred = tok.convert_ids_to_tokens(int(pred_idx))
    return tok.convert_tokens_to_string(pred)


def gpt_predictor(request, n=3):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if request.method == 'GET':
        return "Welcome to GPT predictor"

    if request.method == 'POST':
        data = request.get_json()
        text = data["text"]
        res = []
        n = data["n"]
        for i in range(n):
            pred = get_pred(text, model, tok)
            if pred == "<|endoftext|>":
                break
            else:
                text += pred
        return text