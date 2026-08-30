"""Level-4 banked-pullout solve with the mca_full timestep.

main.py exposes --mca, which selects `mca_state` (mca_include_mudot=False).
The paper's L4 figures need `mca_full`, i.e. the per-action timestep that also
carries |mu_dot|/h_mu, and that flag has no CLI. This mirrors run_pipeline()
with the one field set, so nothing else about the pipeline changes.

The two schemes are defined in analysis/dt_ablation.py:

    mca_state  use_mca_timestep=True, dt_max=0.10, mca_include_mudot=False
    mca_full   use_mca_timestep=True, dt_max=0.10, mca_include_mudot=True

At L3, mca_state hits the 1000-iteration cap with a 3.05e-05 residual while
mca_full closes in 114 iterations at 1.53e-05 and leaves a quarter of the
mu_dot islands, which is why L4 is run with mca_full.

Figures come out prefixed `banked_glider_L4_mca_`; the paper cites four of them
without the `_mca` infix, so they need renaming afterwards.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import RESULTS_DIR, train_or_load_policy
from analysis.experiments import get_setup_for_level, run_profiling
from analysis.plotting import (plot_all_paper_style_policies,
                               plot_value_function_contours,
                               validate_trajectories_with_casadi)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("L4-mca-full")

LEVEL = 4
PREFIX = f"banked_glider_L{LEVEL}_mca"

setup = get_setup_for_level(LEVEL)
env, states, actions, config = setup()

config.use_mca_timestep = True
config.mca_include_mudot = True          # <- this is what makes it mca_full
config.dt_max = 0.10

log.info("=" * 60)
log.info("  Banked Glider - Level %d  [MCA timestep, mca_full]", LEVEL)
log.info("  states=%s  actions=%s", len(states), len(actions))
log.info("  use_mca_timestep=%s  mca_include_mudot=%s  dt_max=%s",
         config.use_mca_timestep, config.mca_include_mudot, config.dt_max)
log.info("=" * 60)

pi, trained = train_or_load_policy(env, states, actions, config, PREFIX,
                                   retrain=True)
log.info("[*] profiling")
run_profiling(setup)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log.info("[*] policy maps")
plot_all_paper_style_policies(pi, PREFIX, show_mesh_lines=None)
log.info("[*] value function contours")
plot_value_function_contours(pi, PREFIX)
log.info("[*] CasADi trajectory validation")
validate_trajectories_with_casadi(pi, PREFIX)
log.info("[+] done -> %s", RESULTS_DIR.resolve())
