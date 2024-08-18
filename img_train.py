import torch
import argparse
import os
from PIL import Image
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import webbrowser
import time

from model import SirenImage
from utils import create_grid, inference_run

parser = argparse.ArgumentParser("siren trainer for images")
parser.add_argument("img_path", type=str, default="images/comet.jpg", help="path to training image")
parser.add_argument("-d", "--device", type=str, default="cuda", help="device to train on")
parser.add_argument("--experiment_name", type=str, default="exp", help="experiment name")
parser.add_argument("-lr", "--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("-e", "--epochs", type=int, default=3000, help="training epochs")
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
experiment_name = args.experiment_name
save_path = os.path.join("models", experiment_name + ".pt")
experiment_path = os.path.join("logs", experiment_name)
max_batch_size = 500 * 500

tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", experiment_path, "--port", "6006"])
time.sleep(5)
webbrowser.open("http://localhost:6006")

img = Image.open(img_path).convert("RGB")
to_tensor = transforms.Compose([
    transforms.Resize(img_height),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
to_tensor_no_norm = transforms.Compose([
    transforms.Resize(img_height),
    transforms.ToTensor()
])
to_image = transforms.ToPILImage()

gt = to_tensor(img).permute(1, 2, 0)
display_gt = to_tensor_no_norm(img)
model_input = create_grid(gt.shape[:-1])

total_size = gt.shape[0] * gt.shape[1]
n_batches = (total_size // max_batch_size) + 1

gt_flat, model_input_flat = gt.flatten(0, 1), model_input.flatten(0, 1)
gt_chunks = torch.chunk(gt_flat, n_batches)
model_input_chunks = torch.chunk(model_input_flat, n_batches)

model = SirenImage(n_layers=n_layers, hidden_ft=hidden_size)
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"Continuing training from checkpoint at {save_path}...")
model = model.to(dev)

opt = optim.Adam(model.parameters(), lr=lr)
crit = nn.MSELoss()

writer = SummaryWriter(experiment_path)
writer.add_image("original", display_gt, 0)

loop = tqdm(range(epochs), total=epochs, desc=f"{n_batches} batches per epoch...", ncols=100)
try:
    for e in loop:
        total_epoch_loss = 0
        for x, y in zip(model_input_chunks, gt_chunks):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            yhat = model(x)
            loss = crit(yhat, y)
            loss.backward()
            opt.step()
            total_epoch_loss += loss.item()
        writer.add_scalar("train_loss", total_epoch_loss / len(model_input_chunks), e)

        if e % 50 == 0:
            torch.save(model.state_dict(), save_path)

            model.eval()
            for i in range(1, 4):
                sample_output = inference_run(model, gt.shape[:-1], i)
                writer.add_image(f"{i}x res", sample_output, e)
            model.train()
except KeyboardInterrupt:
    pass
finally:
    print("Stopping...")
    torch.save(model.state_dict(), save_path)
    tensorboard_process.terminate()

