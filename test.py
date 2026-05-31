import torch


mask = torch.zeros((8, 8)).bool()
mask[:-2, -2:] = True

print(mask)