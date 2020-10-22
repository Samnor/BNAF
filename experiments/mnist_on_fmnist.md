# BNAF trained on mnist evaluated on fmnist

![histogram](mnist_on_fmnist_better.png)

## Hyperparameters

{
    "batch_dim": 200,
    "clip_norm": 0.1,
    "cooldown": 10,
    "dataset": "mnist",
    "decay": 0.5,
    "device": "cuda:0",
    "early_stopping": 100,
    "epochs": 1000,
    "expname": "",
    "flows": 1,
    "hidden_dim": 10,
    "layers": 2,
    "learning_rate": 0.01,
    "load": null,
    "min_lr": 0.0005,
    "n_dims": 784,
    "path": "checkpoint\\mnist_layers2_h10_flows1_gated_2020-10-07-09-46-08",
    "patience": 20,
    "polyak": 0.998,
    "residual": "gated",
    "save": true,
    "tensorboard": "tensorboard"
}
