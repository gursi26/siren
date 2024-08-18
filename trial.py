import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import SirenImage
from utils import create_grid

dev = torch.device("cuda")

def gradient(image: torch.Tensor):
    """
    Computes the gradients of a 3-channel RGB image in the x and y directions.

    Args:
        image (torch.Tensor): A 4D tensor of shape [1, 3, H, W] representing a batch with one RGB image.

    Returns:
        gradient_x (torch.Tensor): The gradient of the image in the x direction.
        gradient_y (torch.Tensor): The gradient of the image in the y direction.
    """
    # Ensure the image is 4D [1, 3, H, W]
    assert image.dim() == 4 and image.size(1) == 3, "Input must be a 4D tensor of shape [1, 3, H, W]."

    # Define Sobel filters for x and y gradients
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1], 
                            [0,  0,  0], 
                            [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Apply the Sobel filters to each channel separately using group convolution
    gradient_x = F.conv2d(image, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    gradient_y = F.conv2d(image, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    return gradient_x, gradient_y

img = Image.open("images/toronto.jpg")
img_tensor = transforms.ToTensor()(img)
dx, dy = gradient(img_tensor.unsqueeze(0))

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 20))
ax = ax.flatten()

ax[0].imshow(dx[0].permute(1, 2, 0))
ax[1].imshow(dy[0].permute(1, 2, 0))


model = SirenImage(n_layers=9)
model.load_state_dict(torch.load("models/1080p-9layer-toronto-mixres.pt", weights_only=True))
model = model.to(dev)
for p in model.parameters():
    p.requires_grad = False

grid = create_grid((img.size[::-1]))
grid_shape = grid.shape[:-1]
grid.requires_grad = True

for chunk in grid.view(-1, 2).chunk(500, dim=0):
    output = model(chunk.to(dev))
    output.sum().backward()

grid_grad = grid.grad

grid_grad
dx
plt.imshow(grid_grad[:, :, 0])

