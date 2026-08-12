"""Ejecuta los generadores de paper_procedures.py que quedaron sin punto de
entrada en su __main__.

Tres artefactos del paper no se podian reproducir desde la CLI:

    ic_gamma_alpha.json  -> fig_ic_gamma_alpha.pdf, fig_ic_optimum.pdf
    ic_montecarlo.json   -> table_montecarlo.tex

Las funciones existen y estan probadas; lo que falta es llamarlas. Este
driver las invoca en el orden correcto (el gmax lee el json que produce el
paso anterior) sin tocar paper_procedures.py.

Uso:
    python regen_orphans.py            # gamma-alpha (rapido) + montecarlo
    python regen_orphans.py --ga       # solo gamma-alpha
    python regen_orphans.py --mc       # solo montecarlo (1000 rollouts/brazo)
"""
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import paper_procedures as pp


# GA_ARMS por defecto trae solo ['optimal', 'caa_ramp', 'gated'], pero
# make_ic_procedures_figure (figura 8 del paper) tambien necesita 'pd'. La
# corrida publicada se lo paso a mano y eso nunca quedo escrito, asi que al
# regenerar la figura fallaba con KeyError: 'pd'. Se pasan los cuatro de una
# sola vez: hacerlo en dos barridos (tres brazos y despues uno) cuesta 2325
# rollouts extra y deja el json en un estado intermedio invalido.
ARMS = ["optimal", "caa_ramp", "gated", "pd"]


def gamma_alpha():
    logger.info(f"[*] compute_ic_gamma_alpha {ARMS} ...")
    pp.compute_ic_gamma_alpha(arms=ARMS)
    logger.info("[*] compute_ic_gamma_alpha_gmax ...")
    pp.compute_ic_gamma_alpha_gmax()
    logger.info("[*] figuras gamma-alpha ...")
    # OJO con los nombres, son faciles de confundir y ya me costo dos veces:
    #   make_ic_optimum_figure   -> fig_ic_optimum          (FIGURA 7 del paper)
    #   make_optimal_ic_figure   -> fig_optimal_ic_heatmap  (otra figura)
    pp.make_ic_gamma_alpha_figure()
    pp.make_ic_optimum_figure()      # fig_ic_optimum      <- figura 7
    pp.make_optimal_ic_figure()      # fig_optimal_ic_heatmap
    pp.make_ic_procedures_figure()   # fig_ic_procedures   <- figura 8
    logger.info("[+] ic_gamma_alpha.json + fig_ic_optimum + fig_ic_procedures")


def montecarlo():
    logger.info("[*] compute_ic_montecarlo (1000 entradas por brazo) ...")
    pp.compute_ic_montecarlo()
    pp.report_ic_montecarlo()
    pp.write_montecarlo_table()
    logger.info("[+] ic_montecarlo.json + table_montecarlo.tex")


if __name__ == "__main__":
    solo_ga = "--ga" in sys.argv
    solo_mc = "--mc" in sys.argv
    if not solo_mc:
        gamma_alpha()
    if not solo_ga:
        montecarlo()
