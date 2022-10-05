# from http.client import _DataType
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim as opt
torch.manual_seed(2)
# word_conversion = {"hey": 0, "there": 1}
# embeddings = nn.Embedding(2, 3)
# lookup = torch.tensor([word_conversion["hey"]], dtype=torch.long)
# hey_embeddings = embeddings(lookup)
# print(hey_embeddings)
# n, d, m = 2, 4, 6 
# embedding = nn.Embedding(n, d)
# embedding_1 = embedding
# print(embedding.weight)
# print(embedding_1( (torch.Tensor([0,1])).int()))
# print(embedding_1.weight)

# a = torch.Tensor([[1,2]])
# b = torch.Tensor([1,2])
# print(a.squeeze())

a = nn.parameter.Parameter(torch.Tensor([0,1]),requires_grad=False)
b=a
a += 1
print(a,b)

# print(a.data_ptr==b.data_ptr)
# print(b.data_ptr)
