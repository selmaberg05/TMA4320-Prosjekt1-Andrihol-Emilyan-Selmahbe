"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    nn_params, loss = train_nn(sensor_data, cfg)
    T_pred = predict_grid(nn_params,x,y,t,cfg)

    #Plotter losses
    plt.figure()
    for key, values in loss.items():
        plt.plot(values, label=key)

    plt.yscale("log")        
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/nn/nn_loss.png", dpi=200) # Lag en mappe "nn" under outputs for plottene
    plt.close()

    # Plotter temperaturen fra NN
    plot_snapshots(x,y,t, T_pred,save_path="output/nn/nn_snapshots.png",)
    create_animation(x, y, t, T_pred, title="NN", save_path="output/nn/nn_animation.gif")
  
    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()

