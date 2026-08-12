"""Agrega el brazo 'pd' a ic_gamma_alpha.json y rehace fig_ic_procedures.

GA_ARMS por defecto trae solo ['optimal', 'caa_ramp', 'gated'], pero
make_ic_procedures_figure espera tambien 'pd' (la corrida publicada lo paso
explicitamente). compute_ic_gamma_alpha PISA el json, asi que hay que
calcular 'pd' aparte y mergear en vez de recalcular los cuatro brazos.
"""
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import paper_procedures as pp

JSON = pp.OUT_DIR / "ic_gamma_alpha.json"


def main():
    completo = json.loads(JSON.read_text())
    if "pd" in completo["arms"]:
        logger.info("[=] 'pd' ya esta; solo rehago las figuras")
    else:
        respaldo = JSON.with_suffix(".json.sin_pd")
        shutil.copy2(JSON, respaldo)
        logger.info(f"[*] respaldo en {respaldo.name}")

        # esto sobrescribe el json con SOLO el brazo pd
        pp.compute_ic_gamma_alpha(arms=["pd"])
        solo_pd = json.loads(JSON.read_text())

        completo["arms"]["pd"] = solo_pd["arms"]["pd"]
        JSON.write_text(json.dumps(completo, indent=1))
        logger.info(f"[+] merge listo; brazos = {list(completo['arms'])}")

    pp.make_ic_procedures_figure()
    pp.make_ic_optimum_figure()
    pp.make_ic_gamma_alpha_figure()
    logger.info("[+] figuras gamma-alpha rehechas")


if __name__ == "__main__":
    main()
