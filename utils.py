import torch
from torchvision import transforms

def create_grid(grid_shape):
    vals = tuple([torch.linspace(-1, 1, gs) for gs in grid_shape])
    grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid(vals, indexing="ij")[::-1]], dim=-1)
    return grid

def inference_run(model, shape, scale=1):
    inference_max_size = 1080 * 720
    grid_y, grid_x = int(shape[0] * scale), int(shape[1] * scale)
    n_batches = ((grid_x * grid_y) // inference_max_size) + 1
    with torch.no_grad():
        grid = create_grid((grid_y, grid_x))
        grid = grid.flatten(0, 1)
        outputs = []
        for grid_input in grid.chunk(n_batches, dim=0):
            yhat = model(grid_input.to(next(model.parameters()).device)).permute(1, 0)
            yhat = ((yhat + 1) / 2).clip(0, 1)
            outputs.append(yhat.detach())
        outputs = torch.cat(outputs, dim=-1)
        return outputs.view(outputs.shape[0], grid_y, grid_x).cpu()