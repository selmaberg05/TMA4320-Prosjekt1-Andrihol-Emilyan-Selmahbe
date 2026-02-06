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
    # Lager mappe under output til animasjon og plot
    os.makedirs("output/nn", exist_ok=True)
    
    # Definerer parametre
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    nn_params, loss = train_nn(sensor_data, cfg)
    T_pred = predict_grid(nn_params,x,y,t,cfg)

    # Plotter tapsfunksjoner
    plt.figure()
    for key, values in loss.items():
        plt.plot(values, label=key)

    plt.yscale("log")  # Bruker logaritmisk skala 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/nn/nn_loss.png", dpi=200) # Legger figurene i en egen mappe i "output" mappen
    plt.close()

    # Plotter den predikerte temperaturen fra det nevrale nettverket 
    plot_snapshots(x,y,t, T_pred,save_path="output/nn/nn_snapshots.png",)
    create_animation(x, y, t, T_pred, title="NN", save_path="output/nn/nn_animation.gif")

    # Plotter differansen mellom temperatur fra numerisk løser og nevralt nettverk
    plot_snapshots(x,y,t, T_fdm-T_pred,save_path="output/nn/nn_vs_numsolver_snapshots.png")
    create_animation(x, y, t, T_fdm-T_pred, title="NN", save_path="output/nn/nn_vs_numsolver_animation.gif")
    
    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
