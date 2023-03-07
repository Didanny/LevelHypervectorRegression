import torch

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
    update = M + 0.00001 * torchhd.bind(x, (embed(y)))
    update = update.mean(0)
    # print(update)
    return update.mean(0)

def forward(x, M):
    l = torchhd.bind(M, torchhd.inverse(encode(x)))
    # print(l)
    l = torchhd.cleanup(l, memory)
    # print(l)
    return ((memory == l).all(dim=1).nonzero().squeeze() / 10).mean(0)

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
