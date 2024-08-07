import torch
import argparse
import math
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SIREN
from utils import create_grid, inference_run

parser = argparse.ArgumentParser("siren trainer for images")
parser.add_argument("img_path", type=str, default="images/comet.jpg", help="path to training image")
parser.add_argument("-e", "--epochs", type=int, default=3000, help="training epochs")
parser.add_argument("-lr", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-d", "--device", type=str, default="cuda", help="device to train on")
parser.add_argument("-i", "--img_height", type=int, default=256, help="image height for train resize")
parser.add_argument("-n", "--n_layers", type=int, default=3, help="number of layers")
parser.add_argument("-s", "--hidden_size", type=int, default=256, help="hidden layer size")
args = parser.parse_args()

img_path = args.img_path
epochs = args.epochs
lr = args.lr
n_layers = args.n_layers
dev = torch.device(args.device)
img_height = args.img_height
hidden_size = args.hidden_size

img = Image.open(img_path).convert("RGB")
to_tensor = transforms.Compose([
    transforms.Resize(img_height),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
to_image = transforms.ToPILImage()

gt = to_tensor(img).to(dev)
model_input = create_grid(gt.shape[1:]).to(dev)

model = SIREN(n_layers=n_layers, hidden_ft=hidden_size).to(dev)
opt = optim.Adam(model.parameters(), lr=lr)
crit = nn.MSELoss()

loop = tqdm(range(epochs), total=epochs, desc="Training model...")
for e in loop:
    model.train()
    opt.zero_grad()
    yhat = model(model_input)
    loss = crit(yhat.permute(2, 0, 1), gt)
    loss.backward()
    opt.step()

    loop.set_postfix(loss=loss.item())

model.eval()
output = inference_run(model, gt.shape[1:], dev)
plt.imshow(output)
plt.show()