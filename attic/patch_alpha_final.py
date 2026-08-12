"""Rellena alpha_final_deg en la IC canonica de robustness.json.

Por que existe este script en vez de reejecutar run_matrix: alpha_final_deg se
agrego a run_matrix despues de haber corrido la matriz, y run_matrix son 441
rollouts (~25 min) porque barre 9 IC por celda. La figura de la matriz
(make_matrix_figure) lee UNICAMENTE la IC canonica, asi que para que las marcas
aparezcan basta recalcular esas 49. El resto de las IC quedan sin el campo hasta
la proxima corrida completa de run_matrix, que ya lo escribe para las nueve.

Uso:  .venv/bin/python patch_alpha_final.py
"""
import json
from pathlib import Path

import numpy as np
from loguru import logger

import paper_robustness as R
from paper_procedures import ctrl_optimal, rollout
from aircraft.symmetric_stall import SymmetricStall
from PolicyIteration import PolicyIterationStall

JSON = R.OUT_DIR / "robustness.json"


def main():
    pi = PolicyIterationStall.load(
        Path("results/SymmetricStall_policy.npz"), env=SymmetricStall())
    data = json.loads(JSON.read_text())
    a0, v0f = R.CANONICAL
    ck = f"a{a0:.0f}_v{v0f:.2f}"

    n_patched = 0
    for mf in R.MASS_FACTORS:
        vnorm0 = v0f * float(np.sqrt(mf))
        for dx in R.DXCG_LIST:
            env = R.perturbed_env(mf, dx)
            r = rollout(env, pi, ctrl_optimal, a0, vnorm0, record=True)
            cell = data["cells"][f"m{mf:.2f}_dx{dx:+.2f}"][ck]
            # sanity: el rollout tiene que reproducir la altura ya almacenada,
            # si no es que el modelo o la politica cambiaron desde la matriz
            if abs(r["h"] - cell["h"]) > 1e-6:
                raise SystemExit(
                    f"m{mf:.2f} dx{dx:+.2f}: h={r['h']:.6f} no reproduce el "
                    f"{cell['h']:.6f} guardado -- la matriz esta desactualizada, "
                    f"hay que reejecutar run_matrix, no parchar")
            cell["alpha_final_deg"] = float(np.rad2deg(r["hist"]["alpha"][-1]))
            n_patched += 1

    JSON.write_text(json.dumps(data, indent=1))
    logger.info(f"[+] alpha_final_deg escrito en {n_patched} celdas ({ck})")

    flagged = [(mf, dx, c) for mf in R.MASS_FACTORS for dx in R.DXCG_LIST
               for c in [data["cells"][f"m{mf:.2f}_dx{dx:+.2f}"][ck]]
               if c["status"] == "recovered" and c["alpha_final_deg"] > 14.0]
    logger.info(f"[+] cierran estancadas: {len(flagged)}")
    for mf, dx, c in flagged:
        logger.info(f"    m{mf:.2f} dx{dx:+.2f}: alpha_f="
                    f"{c['alpha_final_deg']:.1f} deg, h={c['h']:.2f} m, "
                    f"t={c['t']:.2f} s")


if __name__ == "__main__":
    main()
