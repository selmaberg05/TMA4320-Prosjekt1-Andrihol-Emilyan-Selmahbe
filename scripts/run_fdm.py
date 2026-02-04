"""Script for running and plotting the FDM solution."""

from viz import create_animation, plot_snapshots

from project import (
    load_config,
    solve_heat_equation,
)


def main():
    cfg = load_config("config.yaml")

    print("Solving heat equation with FDM...")
    x, y, t, T_fdm = solve_heat_equation(cfg)

    print("\nGenerating FDM visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_fdm,
        save_path="output/fdm/fdm_snapshots.png",
    )
    create_animation(
        x, y, t, T_fdm, title="FDM", save_path="output/fdm/fdm_animation.gif"
    )


if __name__ == "__main__":
    main()
