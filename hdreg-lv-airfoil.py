import torchhd
from torchhd.datasets import AirfoilSelfNoise, BeijingAirQuality
from torchhd import embeddings
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchmetrics
from tqdm import tqdm

DIMENSIONS = 5
BASELINE = False

class BaselineModel(torch.nn.Module):
    def __init__(self, num_features, lr=0.00001) -> None:
        super(BaselineModel, self).__init__()
        
        self.lr = lr
        self.M = torch.zeros(1, DIMENSIONS)
        self.project = embeddings.Projection(num_features, DIMENSIONS)
        
    def encode(self, x):
        sample_hv = self.project(x)
        return torchhd.hard_quantize(sample_hv)
    
    def model_update(self, x, y):
        update = self.M + self.lr * (y - (F.linear(x, self.M))) * x
        update = update.mean(0)
        # print("update: ", update)
        self.M = update
        
    def forward(self, x):
        return F.linear(self.encode(x), self.M)
    
class LevelHVModel(torch.nn.Module):
    def __init__(self, num_features, num_levels=100, lr=0.00001) -> None:
        super(LevelHVModel, self).__init__()
        
        self.lr = lr
        self.M = torch.zeros(1, DIMENSIONS)
        self.project = embeddings.Projection(num_features, DIMENSIONS)
        self.num_levels = num_levels
        self.embed = embeddings.Level(num_levels, DIMENSIONS)
        self.memory = self.embed.weight
        
    def encode(self, x):
        sample_hv = self.project(x)
        return torchhd.hard_quantize(sample_hv)
    
    def model_update(self, x, y):
        self.M = self.M + self.lr * torchhd.bind(x, (self.embed(y)))
        # update = update.mean(0)
        # # print(update)
        # self.M = update
        
    def forward(self, x):
        l = torchhd.bind(self.M, torchhd.inverse(self.encode(x)))
        l = torchhd.hard_quantize(l)
        # input = torchhd.as_vsa_model(l)
        # scores = torchhd.cos_similarity(input, self.memory)
        l = torchhd.cleanup(l, self.memory)
        # try:
        #     l = torchhd.cleanup(l, self.memory)
        # except Exception as e:
        #     print(e)
        #     a = 2
        #     pass
        i = (self.memory == l).all(dim=1).nonzero().squeeze()
        return (((i / self.num_levels) * (self.embed.high - self.embed.low)) + self.embed.low).mean(0)
        try :
            l = torchhd.cleanup(l, self.memory)
            i = (self.memory == l).all(dim=1).nonzero().squeeze()
            return (((i / self.num_levels) * (self.embed.high - self.embed.low)) + self.embed.low).mean(0)
        except:
            i = torch.Tensor([self.num_levels / 2])
            # print((((i / self.num_levels) * (self.embed.high - self.embed.low)) + self.embed.low).mean(0))
            return (((i / self.num_levels) * (self.embed.high - self.embed.low)) + self.embed.low).mean(0)
        # print(l)
    
dataset = AirfoilSelfNoise('../data', download=True)

STD_DEVS = dataset.data.std(0)
MEANS = dataset.data.mean(0)
TARGET_STD = dataset.targets.std(0)
TARGET_MEAN = dataset.targets.mean(0)
MINS = dataset.data.min(0).values
MAXS = dataset.data.max(0).values
TARGET_MINS = dataset.targets.min(0).values
TARGET_MAXS = dataset.targets.max(0).values

def transform(x):
    x = x - MINS
    x = x / (MAXS - MINS)
    return x

def target_transform(x):
    return x
    # x = x - TARGET_MINS
    # x = x / (TARGET_MAXS - TARGET_MINS)
    # return x

# def transform(x):
#     x = x - MEANS
#     x = x / STD_DEVS
#     return x


# def target_transform(x):
#     x = x - TARGET_MEAN
#     x = x / TARGET_STD
#     return x

dataset.transform = transform
dataset.target_transform = target_transform

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1)

if BASELINE:
    model = BaselineModel(5)
else:
    model = LevelHVModel(5, 3, lr=1)

mse = torchmetrics.MeanSquaredError()
samples = []
labels = []
d = iter(train_dataloader)
for i in range(2):
    sample, label = next(d)
    samples.append(sample)
    labels.append(label)
    # sample_hv = model.encode(sample)
    sample_hv = model.project(sample)
    # model.model_update(sample_hv, (label - TARGET_MINS)/(TARGET_MAXS - TARGET_MINS))
    label_norm = (label - TARGET_MINS)/(TARGET_MAXS - TARGET_MINS)
    label_level = model.embed(label_norm)
    model.M += torchhd.bind(sample_hv, model.embed(label_norm))
    
for sample, label in zip(samples, labels):
    prediction = model(sample)
    prediction_f = prediction * (TARGET_MAXS - TARGET_MINS) + TARGET_MINS
    print(prediction_f, label)
    mse.update(prediction_f.view(1).cpu(), label)
print(f"Training mean squared error of {(mse.compute().item()):.3f}")    


# with torch.no_grad():
#     for _ in range(1):
#         for samples, labels in tqdm(train_dataloader, desc="Iteration {}".format(_ + 1)):
#             samples_hv = model.encode(samples)
#             model.model_update(samples_hv, labels)
            

# mse = torchmetrics.MeanSquaredError()

# with torch.no_grad():
#     for samples, labels in tqdm(test_dataloader, desc="Testing"):
#         predictions = model(samples)
#         predictions = predictions * (TARGET_MAXS - TARGET_MINS) + TARGET_MINS
#         # print(predictions)
#         labels = labels
#         # print(labels)
#         mse.update(predictions.view(1).cpu(), labels)

# print(f"Testing mean squared error of {(mse.compute().item()):.3f}")

# with torch.no_grad():
#     for _ in range(20):
#         for samples, labels in tqdm(train_dataloader, desc="Iteration {}".format(_ + 1)):
#             samples_hv = model.encode(samples)
#             model.model_update(samples_hv, labels)
            
# mse = torchmetrics.MeanSquaredError()

# p = []
# l = []
# with torch.no_grad():
#     for samples, labels in tqdm(test_dataloader, desc="Testing"):
#         predictions = model(samples)
#         predictions = predictions * (TARGET_MAXS - TARGET_MINS) + TARGET_MINS
#         labels = labels
#         mse.update(predictions.view(1).cpu(), labels)
#         # print(predictions, labels)

# print(f"Testing mean squared error of {(mse.compute().item()):.3f}")

