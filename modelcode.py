import torch
import torch.nn as nn
from transformers import  BertModel

class SentimentModel(nn.Module):
    def __init__(self, num_classes=5, freeze_bert = True):
        super(SentimentModel, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.final_layer = nn.Linear(768, num_classes)

    def forward(self, seq, attn_masks):

        cont_reps = self.bert_layer(seq, attention_mask = attn_masks)
        cls_rep = cont_reps[0][:,0]
        logits = self.final_layer(cls_rep)
        return logits
    