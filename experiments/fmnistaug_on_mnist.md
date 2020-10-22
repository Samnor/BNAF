# BNAF trained on fmnist with and without rotations evaluated on mnist

![histogram_rotations](rotations.png)

Fmnistaug is all of fmnist train (60 000 images) + all of the fmnist train images rotated randomly with torchvision. So 120 000 images in total.

## Hyperparameters

{
    "batch_dim": 200,
    "clip_norm": 0.1,
    "cooldown": 10,
    "dataset": "fashionmnistaug",
    "decay": 0.5,
    "device": "cuda:0",
    "early_stopping": 100,
    "epochs": 1000,
    "expname": "",
    "flows": 1,
    "hidden_dim": 10,
    "layers": 1,
    "learning_rate": 0.01,
    "load": null,
    "min_lr": 0.0005,
    "n_dims": 784,
    "path": "checkpoint\\fashionmnistaug_layers1_h10_flows1_gated_2020-10-15-13-12-50",
    "patience": 20,
    "polyak": 0.998,
    "residual": "gated",
    "save": true,
    "tensorboard": "tensorboard"
}

![histogram_no_rotations](no_rotations.png)

Above you can see the results when the model was trained on only fmnist train.

## Hyperparameters

{
    "batch_dim": 200,
    "clip_norm": 0.1,
    "cooldown": 10,
    "dataset": "fashionmnist",
    "decay": 0.5,
    "device": "cuda:0",
    "early_stopping": 100,
    "epochs": 1000,
    "expname": "",
    "flows": 1,
    "hidden_dim": 10,
    "layers": 1,
    "learning_rate": 0.01,
    "load": null,
    "min_lr": 0.0005,
    "n_dims": 784,
    "path": "checkpoint\\fashionmnist_layers1_h10_flows1_gated_2020-10-15-09-32-52",
    "patience": 20,
    "polyak": 0.998,
    "residual": "gated",
    "save": true,
    "tensorboard": "tensorboard"
}




