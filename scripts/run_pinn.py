"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)



def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    pinn_params, losses = train_pinn(sensor_data,cfg)
    T_pred = predict_grid(pinn_params["nn"],x,y,t,cfg)

    #Plotter losses
    plt.figure()
    for key, values in losses.items():
        plt.plot(values, label=key)

    plt.yscale("log")        
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/pinn/pinn_loss.png", dpi=200) # Lag en mappe "pinn" under outputs for plottene
    plt.close()

    # Plotter temperaturen fra NN
    plot_snapshots(x,y,t, T_pred,save_path="output/pinn/pinn_snapshots.png",)
    create_animation(x, y, t, T_pred, title="PINN", save_path="output/pinn/pinn_animation.gif")
  
    alpha = float(np.exp(np.array(pinn_params["log_alpha"])[0]))
    k = float(np.exp(np.array(pinn_params["log_k"])[0]))
    h = float(np.exp(np.array(pinn_params["log_h"])[0]))
    power = float(np.exp(np.array(pinn_params["log_power"])[0]))

    print("Learned physical parameters:")
    print(f"alpha = {alpha}")
    print(f"k = {k}")
    print(f"h = {h}")
    print(f"power = {power}")

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
