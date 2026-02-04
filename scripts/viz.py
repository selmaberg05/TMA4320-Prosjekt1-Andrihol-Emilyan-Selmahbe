"""Visualization utilities for PINN results."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_snapshots(
    x,
    y,
    t,
    T,
    title=None,
    cmap="hot",
    save_path="output/snapshots.png",
    show_interactively=False,
):
    """Plot temperature field at 3 different times as subplots.

    Args:
        x, y: Spatial coordinates
        t: Time points
        T: Temperature solution (nt, nx, ny)
        title: Title prefix for the plot
        cmap: Colormap
        save_path: Path to save the figure
        show_interactively: Whether to display the plot interactively
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    vmin, vmax = T.min(), T.max()

    # Select 3 time indices: early, middle, late
    time_indices = [0, len(t) // 2, -1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, idx in zip(axes, time_indices):
        im = ax.pcolormesh(
            X, Y, T[idx], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"t = {t[idx]:.2f} h")
        ax.set_aspect("equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label="Temperature")

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if show_interactively:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


def create_animation(
    x,
    y,
    t,
    T,
    title="Temperature",
    cmap="hot",
    save_path="output/animation.gif",
    fps=10,
):
    """Create animation of temperature field over time.

    Args:
        x, y: Spatial coordinates
        t: Time points
        T: Temperature solution (nt, nx, ny)
        title: Title for the animation
        cmap: Colormap
        save_path: Path to save the animation
        fps: Frames per second
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x, y, indexing="ij")

    vmin, vmax = T.min(), T.max()
    im = ax.pcolormesh(X, Y, T[0], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Temperature")

    time_text = ax.set_title(f"{title}, t = {t[0]:.2f} h")

    def update(frame):
        im.set_array(T[frame].ravel())
        time_text.set_text(f"{title}, t = {t[frame]:.2f} h")
        return [im, time_text]

    print("\nCreating animation...")
    anim = FuncAnimation(fig, update, frames=len(t), interval=1000 / fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved: {save_path}")
