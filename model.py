import torch.nn as nn
from transformers import BertModel

class BertRegressor(nn.Module):
    def __init__(self, drop_rate = 0.2, freeze_bert = False):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        d_in = 768
        d_out = 1
        self.regression_layer = nn.Sequential(

            nn.Dropout(drop_rate),
            nn.Linear(d_in, d_out)
        )
    def forward(self, X, masks):
        output = self.bert(X, masks)
        return self.regression_layer(output[1])