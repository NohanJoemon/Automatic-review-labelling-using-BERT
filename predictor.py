import pandas as pd
import numpy as np
import joblib
import sklearn
import os
import preprocess
import torch
from modelcode import SentimentModel

def predict(text,wpath):
    seq,attn_mask = preprocess.preprocess(text)
    model=SentimentModel()
    model.load_state_dict(torch.load(wpath,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        logits = model(seq[:,0,:], attn_mask[:,0,:])
    pred = logits.max(dim = 1)[1]
    return pred.numpy()[0]+1

