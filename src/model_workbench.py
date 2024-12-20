import torch
import torch.nn as nn
import torch.nn.functional as F
from denoisers.ConditionalUnetMatrixDenoiser import ConditionalUnetMatrixDenoiser
from modules.CrossAttention import CrossAttention


if __name__ == "__main__":
    model = ConditionalUnetMatrixDenoiser(20, 20, 6000, 608).to("cuda")
    x = torch.randn(5, 20, 6000).to("cuda")
    y = torch.randn(5, 20, 6000).to("cuda")
    t = torch.randint(low=1, high=200, size=(5,)).to("cuda")
    M = torch.randn(20, 608, 608).to("cuda")
    m2 = torch.randn(5, 128, 5700).to("cuda")
    x2y2 = torch.randn(5, 128, 3000).to("cuda")
    cams1 = CrossAttention(128, 5700).to("cuda")
    casm1 = CrossAttention(128, 3000).to("cuda")
    res = cams1(m2, x2y2, x2y2)
    res2 = casm1(x2y2, m2, m2)
    print("something")
    # o, m = model(x, t, y)
