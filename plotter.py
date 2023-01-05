import toml
import touchstone

from matplotlib import pyplot as plt

from pathlib import Path
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("Plotter")

    # Chargement des paramètres
    log.info("Chargement des paramètres depuis le toml")
    config = dict()
    with open("RO3210/config.toml", "r") as fhandle:
        config = toml.load(fhandle)

    output_directory = Path("RO3210_output")

    for sim_id, sim_info in config["input"]["files"].items():
        log.info(f"------------------ Process {sim_id}")
        sim_path         = output_directory / sim_id
        infos_path       = sim_path / "infos.toml"

        # Chargement des paramètres de sortie
        log.info("Chargement des paramètres de sortie")
        sim_out_settings = None
        with open(infos_path) as fhandle:
            sim_out_settings = toml.load(fhandle)

        path_s2p_measure = sim_path / sim_out_settings["s2p"]["measure"  ]
        path_s2p_simu    = sim_path / sim_out_settings["s2p"]["simulated"]

        er               = sim_out_settings["parameters"]["er"  ]
        tand             = sim_out_settings["parameters"]["tand"]

        # Chargement des fichiers s2p
        log.info("Chargement des fichiers s2p")
        fmin_mes, vmin_mes, freqs_mes, s21_mes = touchstone.s21_min(path_s2p_measure)
        fmin_sim, vmin_sim, freqs_sim, s21_sim = touchstone.s21_min(path_s2p_simu   )

        # Dessin de la figure
        log.info("Dessin de la figure")
        fig              = plt.figure()
        ax               = fig.add_subplot(1,1,1)

        ax.plot(freqs_mes, s21_mes, linestyle="solid",  color="red"  , label="Mesure"    )
        ax.plot(freqs_sim, s21_sim, linestyle="dotted", color="black", label="Simulation")

        ax.set_xlabel("Freqs [GHz]")
        ax.set_ylabel("$S_{21}$ [dB]")
        ax.grid()
        ax.legend()

        ax.set_title(f"Comparaison pour $\\varepsilon_r = {er}$, $\\mathrm{{tan}}(\\delta) = {tand}$")

        plt.savefig(str(sim_path / "comparaison.png"))