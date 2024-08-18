# SIREN pytorch
A PyTorch implementation of "Implicit Neural Representations with Periodic Activation Functions" by Sitzmann et al.
[Paper](https://arxiv.org/abs/2006.09661) </br>

# Usage
```
❯ python img_train.py -h
usage: siren trainer for images [-h] [-d DEVICE] [--experiment_name EXPERIMENT_NAME] [-lr LR] [-e EPOCHS]
                                [-i IMG_HEIGHT] [-n N_LAYERS] [-s HIDDEN_SIZE]
                                img_path

positional arguments:
  img_path              path to training image

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        device to train on
  --experiment_name EXPERIMENT_NAME
                        experiment name
  -lr LR, --lr LR       learning rate
  -e EPOCHS, --epochs EPOCHS
                        training epochs
  -i IMG_HEIGHT, --img_height IMG_HEIGHT
                        image height for train resize
  -n N_LAYERS, --n_layers N_LAYERS
                        number of layers
  -s HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        hidden layer size
```

# Results
Results after training for 10k epochs on `images/toronto.jpg` with a resolution of 1080p.

| Ground truth    | Model generated |
| -------- | ------- |
| ![Ground truth](images/original-toronto.png)  | ![Model generated](images/model-generated.png)    |
