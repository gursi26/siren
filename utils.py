import torch
from torchvision import transforms

def create_grid(grid_shape):
    vals = tuple([torch.linspace(-1, 1, gs) for gs in grid_shape])
    grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid(vals, indexing="ij")[::-1]], dim=-1)
    return grid

def inference_run(model, shape, dev, scale=1):
    with torch.no_grad():
        grid = create_grid((int(shape[0] * scale), int(shape[1] * scale))).to(dev)
        yhat = model(grid).permute(2, 0, 1)
        yhat = ((yhat + 1) / 2).clip(0, 1)
        return transforms.ToPILImage()(yhat)