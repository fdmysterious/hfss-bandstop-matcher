"""
=============================
S2P processor
=============================

Joséphine et Florian Dupeyron
Octobre 2022

Ce script permet d'identifier les paramètres d'un échantillon sur un capteur CSRR,
en fonction d'une mesure s2p en entrée, et d'une configuration présente dans le fichier
config.toml.
"""

import toml
import logging
import argparse
import traceback

import time

import touchstone
import numpy as np
import os
import sys

from pathlib import Path
from pyaedt  import Desktop, Hfss

# ------------------------- Utilitaires divers

def aedt_cleanup(project_path: Path, log):
    """
    Cette fonction fait un peu de nettoyage dans les dossiers du projet en cas de pépin!
    Attention, c'est un peu sale, et ça ferme toutes les instances de HFSS en cours.
    """
    project_path = Path(project_path)

    log.info(f"Cleanup for project {project_path}")

    log.info(f"    Cleanup ansysedt.exe processes")
    os.system("taskkill /f /im ansysedt.exe")
    time.sleep(0.5)

    log.info(f"    Cleanup .semaphore files")
    for fpath in Path(f"{project_path}results").glob("*.semaphore"):
        fpath.unlink() # Delete 
        
    log.info("f    Remove lock file for project")
    Path(f"{project_path}.lock").unlink(missing_ok=True)


# ------------------------- Fonctions de traitement

def midr(a,b):
    return (a+b)/2

def dist_abs(a,b):
    return np.abs(a-b)

def simu_s21_min(freqs, s21_dB):
    """
    Extraction du minimum à partir des résultats de simulation
    """
    freqs  = np.array(freqs)
    s21_dB = np.array(s21_dB)
    
    print(s21_dB.shape)
	
	# Limit axes for research to 2GHz
    #i_end = np.where(freqs >= 2.0)[0][0]
	
	# Find min
    v_min = np.min(s21_dB)
    i_min = np.where(s21_dB == v_min)[0][0]
    f_min = freqs[i_min]
    
    return f_min, v_min

# ------------------------- Gestion de la simulation

class Simulation_Manager:
    """
    Classe additionnnelle qui fournit des aides
    à la simulation
    """
    def __init__(self, desktop, hfss, config):
        self.desktop    = desktop
        self.hfss       = hfss

        self.setup_name = "sweep_setup"

        self.log        = logging.getLogger("Simulation manager")

    def simulate_for_settings(self, tsample: float, er: float, tand: float, f_min, f_max, npoints):
        """
        Effectue une simulation avec les paramètres fournis
        :param er: Permitivité relative
        :param tand: Pertes
        :return: les fréquences simulées et le tableau S21
        """

        # Set design variables
        self.hfss["$Ers"]     = er
        self.hfss["$tands"]   = tand
        self.hfss["$tsample"] = tsample

        # Create setup
        self.setup = self.hfss.create_setup(self.setup_name)
        
        if not self.setup:
            raise RuntimeError("Failed to create setup")

        self.sweep = self.hfss.create_linear_count_sweep(setupname=self.setup_name, unit="GHz",
            sweepname          = "Sweep",
            sweep_type          = "Interpolating",
            freqstart          = f_min,
            freqstop           = f_max,

            num_of_freq_points = npoints,
            save_fields        = False
        )

        if not self.sweep:
            raise RuntimeError("Failed to create sweep")

        # Save project
        #self.hfss.save_project()

        # Analyze setup
        success = hfss.analyze_setup(self.setup_name)
        if not success:
            raise RuntimeError("Simulation failed")


    def extract_s21(self):
        self.log.info("Extraction des résultats de simulation")
        #report = hfss.post.reports_by_category.standard("S(2,1)")
        #report.create()

        # solution = report.get_solution_data()

        solution = self.hfss.post.get_solution_data("S(2:1,1:1)")
        y_db20   = solution.data_db20()
        x_GHz    = solution.variation_values("Freq") 

        return x_GHz, y_db20


    def save_s21(self, file_path):
        self.hfss.export_touchstone(file_name=str(file_path), sweep_name=self.sweep.name)


    def delete_setup(self):
        self.hfss.delete_setup(self.setup_name)


    def evaluate_side(self, name, target, param, t_sample, ers, tand, f_min, f_max, npoints):
        """
        param permet de déterminer quel paramètre on cherche à évaluer:
            - "freq": Distance vis à vis de la fréquence
            - "s21" : Distance vis à vis du S21
        """
        log.info(f"✍ Evaluation {name}")

        # Simulate
        self.simulate_for_settings(t_sample, ers, tand, f_min, f_max, npoints)

        # Get results
        f_GHz, s21_dB = self.extract_s21()

        # Find minimum
        vf_min, vs21_min = simu_s21_min(f_GHz, s21_dB)
        self.log.info(f" ✔ f_min    = {vf_min:.3f} GHz" )
        self.log.info(f" ✔ S21_min  = {vs21_min:.3f} dB")

        # Target from distance
        distance = None
        if param == "freq":
            distance = dist_abs(vf_min, target)*1000.0
            self.log.info(f" ✔ distance = {distance:.3f} MHz" )
        elif param == "s21":
            distance = dist_abs(vs21_min, target)
            self.log.info(f" ✔ distance = {distance:.3f} dB" )
        else:
            raise ValueError(f"Paramètre inconnu: {param}")


        # Delete setup
        self.delete_setup()

        return distance, vf_min, vs21_min, f_GHz, s21_dB

# ------------------------- Fonctions utilitaires pour les arguments

def arg_folder(xx):
    tt = Path(xx).resolve()
    if tt.exists() and not tt.is_dir():
        raise FileExistsError(f"{tt} is not a regular folder or symlink to a folder")

    return tt

# ------------------------- Programme principal

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("S2P processor")

    # Prise en compte des arguments passés en ligne de commande
    parser = argparse.ArgumentParser(description="CSRR S2P Processor, Joséphine & Florian Dupeyron, October 2022")
    parser.add_argument("input_folder" , type=arg_folder, help="Chemin vers le dossier source")
    parser.add_argument("output_folder", type=arg_folder, help="Chemin vers le dossier de sortie")
    args   = parser.parse_args()


    # Chargement des paramètres depuis le toml
    log.info("Chargement des paramètres depuis le toml")

    config = dict()
    with open(args.input_folder / "config.toml", "r") as fhandle:
        config = toml.load(fhandle)

    # Création du répertoire de sortie
    log.info(f"Création du répertoire de sortie: {args.output_folder}")
    args.output_folder.mkdir(parents=True, exist_ok=True)

    # Chargement des paramètres globaux
    tolerance_mhz = config["goal"]["tolerance_freq"]
    tolerance_db  = config["goal"]["tolerance_s21" ]

    # Chargement paramètres de sortie
    output_fmin, output_fmax = config["output"]["f_range"]
    output_npoints           = config["output"]["npoints"]

    # Ouvture du projet HFSS
    log.info("Ouverture HFSS")

    try:
        with Desktop(specified_version="2022.1", non_graphical=True, close_on_exit=True, new_desktop_session=True, student_version=False) as desktop:
            # Instanciate HFSS, open project and design
            hfss = Hfss(projectname=str(args.input_folder / config["input"]["project_file"]), designname=config["input"]["design_name"])

            # Instanciate simulation manager
            sim_manager = Simulation_Manager(desktop, hfss, config)

            invalidate  = False # True if some parameter has been computed

            for sim_id, sim_info in config["input"]["files"].items():
                file_path = args.input_folder / sim_info["path"]
                log.info(f"-------------- Processing {sim_id} for file {file_path}")

                ers_min , ers_max  = sim_info["ers_range" ]
                tand_min, tand_max = sim_info["tand_range"]
                t_sample           = sim_info["tsample"   ]


                ############################
                # Valeurs cibles
                ############################
                log.info("🔎 Mesure des paramètres cibles")
                target_freq, target_s21, _ , _ = touchstone.s21_min(file_path)

                log.info(f" 🎯 f = {target_freq:.3f} GHz, S21 = {target_s21}dB")

                er              = None # Found Er
                tand            = None # Found tan(delta)
                f_GHz           = None # Frequencies for result s21
                s21_dB          = None # Result s21 curve

                f_min, f_max    = config["input"]["f_range"     ]
                npoints         = config["input"]["npoints"     ]
                npoints_tand    = config["input"]["npoints_tand"]
                target_distance = None
                
                result_data     = dict() # Dictionnaire des données en sortie


                ############################
                # Résultats déjà existants
                ############################
                (args.output_folder / sim_id).mkdir(exist_ok=True, parents=True)

                path_infos = args.output_folder / sim_id / "infos.toml"
                if path_infos.exists() and path_infos.is_file():
                    try:
                        with open(path_infos) as fhandle:
                            result_data = toml.load(fhandle)
                        
                        er = result_data.get("parameters", dict()).get("er", None)
                        if er is not None:
                            log.info("💥 εr déjà disponible, on l'enregistre")

                        tand = result_data.get("parameters", dict()).get("tand", None)
                        if tand is not None:
                            log.info("💥 tan(delta) déjà disponible, on l'enregistre")

                    except Exception as exc:
                        log.warn(f"Erreur lors de l'ouverture du fichier de résultat: {exc}, on ignore")
                        log.debug(traceback.format_exc())


                ############################
                # Save input parameters
                ############################
                log.info("😎 Enregistrement des paramètres d'entrée")
                result_data["parameters"] = result_data.get("parameters", dict())
                result_data["parameters"]["tsample"] = t_sample

                with open(path_infos, "w") as fhandle:
                    toml.dump(result_data, fhandle)


                ############################
                # Dichotomy for Ers
                ############################
                if config["goal"].get("use_found_er", False) and (er is not None):
                    log.info(f"🔥 On utilise la valeur de εr déjà enregistrée")
                else:
                    log.info("🔥 Dichotomie sur εr")
                    invalidate = True

                    ers_bounds   = (ers_min, midr(ers_min, ers_max), ers_max)
                    tand_middle  = midr(tand_min, tand_max)

                    i = 0
                    while True:
                        i += 1
                        log.info(f"────────────── Iteration {i} ──────────────")

                        ers_values   = (midr(ers_bounds[0], ers_bounds[1]), midr(ers_bounds[1], ers_bounds[2]))
                        er_left, er_right = ers_values

                        # Evaluate sides
                        distance_left, vf_min_left, vs21_min_left, f_GHz_left, s21_dB_left = sim_manager.evaluate_side(
                            "plage inférieure",
                            target_freq,
                            "freq",
                            t_sample,
                            er_left,
                            tand_middle,
                            f_min,
                            f_max,
                            npoints
                        )

                        distance_right, vf_min_right, vs21_min_right, f_GHz_right, s21_dB_right = sim_manager.evaluate_side(
                            "plage supérieure",
                            target_freq,
                            "freq",
                            t_sample,
                            er_right,
                            tand_middle,
                            f_min,
                            f_max,
                            npoints
                        )

                        # See wich one wins
                        if distance_right < distance_left:
                            log.info("Right wins!")
                            target_distance = distance_right
                            er    = er_right
                            f_GHz = f_GHz_right
                            s21_dB = s21_dB_right

                            ers_bounds = (ers_bounds[1], er_right, ers_bounds[2])

                            fdist      = abs(vf_min_right-vf_min_left)
                            f_min      = vf_min_right-fdist
                            f_max      = vf_min_right+fdist
                            
                        elif distance_left < distance_right:
                            log.info("Left wins!")
                            target_distance = distance_left
                            er    = er_left
                            f_GHz = f_GHz_left
                            s21_dB = s21_dB_left

                            ers_bounds = (ers_bounds[0], er_left, ers_bounds[1])

                            fdist      = abs(vf_min_right - vf_min_left)
                            f_min      = vf_min_left  - fdist
                            f_max      = vf_min_left  + fdist

                        else:
                            log.info("Convergence! Compare distance from target tan(delta) and break loop")

                            d_left  = dist_abs(target_s21, vs21_min_left )
                            d_right = dist_abs(target_s21, vs21_min_right)

                            if d_left <= d_right:
                                log.info("Left wins!")
                                target_distance = distance_left
                                er              = er_left
                                f_GHz           = f_GHz_left
                                s21_dB          = s21_dB_left

                                fdist           = abs(vf_min_right - vf_min_left)
                                f_min           = vf_min_left  - fdist
                                f_max           = vf_min_left  + fdist

                            else:
                                log.info("Right wins!")
                                target_distance = distance_right
                                er              = er_right
                                f_GHz           = f_GHz_right
                                s21_dB          = s21_dB_right

                                fdist           = (vf_min_right - vf_min_left)
                                f_min           = vf_min_right - fdist
                                f_max           = vf_min_right + fdist

                            break # Break for dichotomy loop

                        if target_distance < tolerance_mhz:
                            break # Break from loop, we have a result!

                    log.info(f" ✔ εr = {er}")

                    # Enregistrement dans le fichier infos
                    log.info("Enregistrement dans le fichier infos")

                    result_data["parameters"] = result_data.get("parameters", dict())
                    result_data["parameters"]["er"] = er
                    with open(path_infos, "w") as fhandle:
                        toml.dump(result_data, fhandle)

                ############################
                # Dichotomy for tand
                ############################
                if config["goal"].get("use_found_tand", False) and (tand is not None):
                    log.info(f"🔥 tan(delta) déjà trouvé !,  tan(delta) = {tand}")
                else:
                    log.info("🔥 Dichotomie sur tan(delta)")
                    invalidate = True

                    tand_bounds  = (tand_min, midr(tand_min, tand_max), tand_max)

                    f_min = target_freq - 0.1
                    f_max = target_freq + 0.1


                    i = 0
                    while True:
                        i += 1
                        log.info(f"────────────── Iteration {i} ──────────────")

                        tand_values   = (midr(tand_bounds[0], tand_bounds[1]), midr(tand_bounds[1], tand_bounds[2]))
                        tand_left, tand_right = tand_values

                        # Evaluate sides
                        distance_left, vf_min_left, vs21_min_left, f_GHz_left, s21_dB_left = sim_manager.evaluate_side(
                            "plage inférieure",
                            target_s21,
                            "s21",
                            t_sample,
                            er,
                            tand_left,
                            f_min,
                            f_max,
                            npoints_tand
                        )

                        distance_right, vf_min_right, vs21_min_right, f_GHz_right, s21_dB_right = sim_manager.evaluate_side(
                            "plage supérieure",
                            target_s21,
                            "s21",
                            t_sample,
                            er,
                            tand_right,
                            f_min,
                            f_max,
                            npoints_tand
                        )

                        # See wich one wins
                        if distance_right < distance_left:
                            log.info("Right wins!")
                            target_distance = distance_right
                            tand  = tand_right
                            f_GHz = f_GHz_right
                            s21_dB = s21_dB_right

                            tand_bounds = (tand_bounds[1], tand_right, tand_bounds[2])
                            
                        elif distance_left < distance_right:
                            log.info("Left wins!")
                            target_distance = distance_left
                            tand  = tand_left
                            f_GHz = f_GHz_left
                            s21_dB = s21_dB_left

                            tand_bounds = (tand_bounds[0], tand_left, tand_bounds[1])

                        else:
                            log.info("Convergence! Compare distance from target tan(delta) and break loop")

                            d_left  = dist_abs(target_s21, vs21_min_left )
                            d_right = dist_abs(target_s21, vs21_min_right)

                            if d_left <= d_right:
                                log.info("Left wins!")
                                target_distance = distance_left
                                tand            = tand_left
                                f_GHz           = f_GHz_left
                                s21_dB          = s21_dB_left

                            else:
                                log.info("Right wins!")
                                target_distance = distance_right
                                tand            = tand_right
                                f_GHz           = f_GHz_right
                                s21_dB          = s21_dB_right

                            break # Break for dichotomy loop

                        if target_distance < tolerance_db:
                            break # Break from loop, we have a result!

                    log.info(f" ✔ tan(delta) = {tand}")

                    # Enregistrement dans le fichier infos
                    log.info("Enregistrement dans le fichier infos")
                    result_data["parameters"]         = result_data.get("parameters", dict())
                    result_data["parameters"]["tand"] = tand
                    with open(path_infos, "w") as fhandle:
                        toml.dump(result_data, fhandle)


                ############################
                # Save results
                ############################
                log.info("💾 Enregistrement des résultats")
                result_data["s2p"] = result_data.get("s2p", dict())
                if(result_data["s2p"].get("done", False) and (not invalidate)):
                    log.info("💾 Déjà effectué!")
                else:
                    target_folder = args.output_folder / sim_id
                    target_folder.mkdir(exist_ok=True)

                    target_file = target_folder / "input.s2p"

                    # Copy file
                    log.info(" -> Copie fichier s2p d'origine")
                    target_file.write_bytes(file_path.read_bytes())

                    log.info(" -> Simulation avec les paramètres finaux")

                    output_s2p = target_folder / "simulated.s2p"

                    sim_manager.simulate_for_settings(t_sample, er, tand, output_fmin, output_fmax, output_npoints)
                    sim_manager.save_s21(output_s2p)
                    sim_manager.delete_setup()

                    # Enregistrement des chemins vers les fichiers de résultat
                    result_data["s2p"]              = result_data.get("s2p", dict())
                    result_data["s2p"]["measure"]   = "input.s2p"
                    result_data["s2p"]["simulated"] = "simulated.s2p"
                    result_data["s2p"]["done"]      = True

                    with open(path_infos, "w") as fhandle:
                        toml.dump(result_data, fhandle)


    except Exception as exc:
        aedt_cleanup(project_path=args.input_folder / config["input"]["project_file"], log=log)
        raise exc