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
    # Lager en mappe under output for plott og animasjon
    os.makedirs("output/pinn", exist_ok=True)

    # Henter ut parametre til PINN
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)
    pinn_params, losses = train_pinn(sensor_data,cfg)
    T_pred_pinn = predict_grid(pinn_params["nn"],x,y,t,cfg)

    # Henter ut parametre til NN for å sammenligne i refleksjonsspørsmålene
    nn_params, losses_nn = train_nn(sensor_data, cfg)
    T_pred_nn = predict_grid(nn_params,x,y,t,cfg)

    # Plotter tapsfunksjonene
    plt.figure()
    for key, values in losses.items():
        plt.plot(values, label=key)

    plt.yscale("log")        
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/pinn/pinn_loss.png", dpi=200) # Lagrer figurene i en mappe "pinn" under outputs
    plt.close()

    # Plotter temperaturen fra PINN
    plot_snapshots(x,y,t, T_pred_pinn,save_path="output/pinn/pinn_snapshots.png",)
    create_animation(x, y, t, T_pred_pinn, title="PINN", save_path="output/pinn/pinn_animation.gif")

    # Plotter differansen i temperatur mellom den numeriske løseren og PINN til refleksjonsspørsmål
    plot_snapshots(x,y,t, (T_fdm-T_pred_pinn),save_path="output/pinn/pinn_vs_numsolution_snapshots.png",)
    create_animation(x, y, t, (T_fdm-T_pred_pinn), title="PINN", save_path="output/pinn/pinn_vs_numsolution_animation.gif")

    # Plotter differansen i temperatur mellom NN og PINN til refleksjonsspørsmål
    plot_snapshots(x,y,t, (T_pred_nn-T_pred_pinn),save_path="output/pinn/pinn_vs_nn_snapshots.png",)
    create_animation(x, y, t, (T_pred_nn-T_pred_pinn), title="PINN", save_path="output/pinn/pinn_vs_nn_animation.gif")

    # Henter og skriver ut de fysiske parametrene
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
