import torch
import torch.nn.functional as F

import torchhd
from torchhd.datasets import AirfoilSelfNoise
from torchhd import embeddings
DIMENSIONS = 10

embed = embeddings.Level(10, DIMENSIONS)
memory = embed.weight
project = embeddings.Projection(1, DIMENSIONS)

data = torch.rand(50).sort().values
targets = torch.cat((torch.zeros(25), torch.ones(25)))

# print(data)
# print(targets)

# print(embed.weight[0].shape)
# hv = embed(torch.Tensor([0.95]))
# hv_1 = embed(torch.Tensor([1.]))
# print((hv == hv_1).sum())

M = torch.zeros(DIMENSIONS)
# print(M)

def encode(x):
    sample_hv = project(x)
    return torchhd.hard_quantize(sample_hv)

def model_update(x, y, M):
    update = M + 0.00001 * (y - (F.linear(x, M))) * x
    update = update
    # print("update: ", update)
    return update

def forward(x, M):
    return F.linear(encode(x), M)

# class BaselineModel(torch.nn.Module):
#     def __init__(self, num_features, lr=0.00001) -> None:
#         super(BaselineModel, self).__init__()
        
#         self.lr = lr
#         self.M = torch.zeros(1, DIMENSIONS)
#         self.project = embeddings.Projection(num_features, DIMENSIONS)
        
#     def encode(self, x):
#         sample_hv = self.project(x)
#         return torchhd.hard_quantize(sample_hv)
    
#     def model_update(self, x, y):
#         update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
#         update = update.mean(0)
#         # print("update: ", update)
#         self.M = update
        
#     def forward(self, x):
#         return F.linear(self.encode(x), self.M)

# print(data.shape)
# print(targets.shape)

# for x in data:
    # print(x.view(1).shape)
    # print(encode(x.view(1)))

print(M)
for i in range(100):
    for x, y in zip(data, targets):
        sample = encode(x.view(1))
        M = model_update(sample, y, M)
    
print(M)
for x in data:
    p = forward(x.view(1), M)
    # print(p)
    # print(memory)
    # print((memory == p))
    print(p)
