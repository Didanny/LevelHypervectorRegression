import math
import torch
import torchhd


def rademacher(*size, dtype=None, device=None, requires_grad=False, generator=None):
    select = torch.empty(size, dtype=torch.bool, device=device)
    select.bernoulli_(generator=generator)

    output = torch.where(select, -1, +1).to(dtype=dtype, device=device)
    output.requires_grad = requires_grad
    return output


def spherical(*size, dtype=None, device=None, requires_grad=False, generator=None):
    output = torch.normal(0, 1, size, dtype=dtype, device=device, generator=generator)
    norm = torch.linalg.norm(output, dim=-1, keepdims=True)
    output.divide_(norm)
    output.requires_grad = requires_grad
    return output


class Levels(torch.nn.Module):
    def __init__(self, num_orthos, dimensions, min=0.0, max=1.0, alpha=1.0) -> None:
        super().__init__()

        assert alpha >= 0.0 and alpha <= 1.0

        self.min_value = min
        self.max_value = max

        self.alpha = alpha
        self.num_orthos = num_orthos
        self.dimensions = dimensions

        self.filter = torch.rand(num_orthos - 1, dimensions)
        self.weight = rademacher(num_orthos, dimensions)

    def forward(self, input):
        shape = input.shape
        out_shape = shape + (self.dimensions,)

        flatview = input.view(-1)
        # map each input scalar value from the input domain's min, max
        # to the number of orthogonal vectors
        max_range = self.num_orthos - 2 + self.alpha
        value = torchhd.map_range(flatview, self.min_value, self.max_value, 0, max_range)
        value = torch.clamp(value, min=0, max=max_range)

        # get the indices of the vectors to interpolate
        # when the value is a round number they will be the same vector 
        # that's OK for the intermediate vectors, but not for the last one
        # because it will go out of bounds
        start_value = value.floor().clamp_max(self.num_orthos - 2)
        start_idx = start_value.long()
        end_idx = value.ceil().long()

        # print(start_idx)
        # print(end_idx)

        # select the 
        filter = self.filter[start_idx]
        start = self.weight[start_idx]
        end = self.weight[end_idx]

        # only consider the value within the interval
        value = value.subtract(start_value)
        # print(value)
        output = torch.where(value.unsqueeze(-1) <= filter, start, end)
        return output.view(out_shape)



if __name__ == "__main__":
    emb = Levels(5, 10000)

    x = torch.linspace(0, 1, 10)
    y = emb(x)

    print(torchhd.cosine_similarity(y, emb.weight))
