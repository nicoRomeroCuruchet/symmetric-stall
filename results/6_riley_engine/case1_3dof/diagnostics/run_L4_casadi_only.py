"""CasADi trajectory validation for the L4 mca_full policy, and nothing else.

The full pipeline died at 29.6 GB of host RAM inside
plot_value_function_contours, which materialises the whole 31.9 M-state grid to
draw contours. That figure is not one the paper uses at L4, and the solve it
depends on is already saved, so this reloads the trained policy and runs only
the validation, which works on trajectories rather than on the grid.

Nothing is retrained: train_or_load_policy with retrain=False reads
results/ReducedBankedGliderPullout_policy.npz, the 255 MB L4 policy.
"""
import logging
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import RESULTS_DIR, train_or_load_policy
from analysis.experiments import get_setup_for_level
from analysis.plotting import validate_trajectories_with_casadi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("L4-casadi")

LEVEL = 4
PREFIX = f"banked_glider_L{LEVEL}_mca"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


env, states, actions, config = get_setup_for_level(LEVEL)()
config.use_mca_timestep = True
config.mca_include_mudot = True
config.dt_max = 0.10
log.info("states=%s  peak RSS after setup: %.1f GB", len(states), rss_gb())

pi, trained = train_or_load_policy(env, states, actions, config, PREFIX,
                                   retrain=False)
if trained:
    log.error("[-] it RETRAINED instead of loading; aborting to avoid a "
              "second 31-minute solve on a policy that already exists")
    sys.exit(2)
log.info("policy loaded. peak RSS: %.1f GB", rss_gb())

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log.info("[*] CasADi trajectory validation -> %s_validation_guided_Fig{3,4}.png",
         PREFIX)
validate_trajectories_with_casadi(pi, PREFIX)
log.info("[+] done. peak RSS: %.1f GB", rss_gb())
