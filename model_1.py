import torch
import torch.nn as nn
import numpy as np

class ad_transformer(nn.Module):
  def __init__(self, d_model, nhead, nlayer, max_token, embedding, device = "cuda"):
    super().__init__()

    self.d_model = d_model
    self.max_token = max_token

    self.embedding = torch.tensor(embedding)

    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, device = device)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer).to("cuda")
    self.linear_out = nn.Linear(d_model * max_token, 2)

  def forward(self, input_str, device = "cuda"):

    #if type(input_str) != list:
    #    raise TypeError("Self Error : Invalid Input !!!")
      
    input = input_str["input_ids"]

    ## Tokens to embeding
    input_embd = torch.zeros(len(input), self.max_token, self.d_model)
    
    input_embd = input_embd + self.embedding[0]   ### Padding
      
    for batch_num in range(len(input)):
        for i in range(len(input[batch_num])):
            input_embd[batch_num][i] = self.embedding[input[batch_num][i]]
    input_embd = input_embd.to(device)
    
    out1 = self.transformer_encoder(input_embd)
    return self.linear_out(out1.reshape(len(out1), self.max_token * self.d_model))