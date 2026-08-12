# refs — bibliography

The PDFs are on disk but **out of git** (copyrighted; see `.gitignore`). This
list is the record of what is there.

| file | reference |
|---|---|
| `Riley.pdf` | Riley, D. R. (1985). Aerodynamic tables for the Grumman AA-1. **The source of the model**: Table I (mass, inertias, geometry), Table III (coefficients vs α for C_T = 0 and 0.5), Appendix A eq. (A4) (first-order engine lag), Appendix B (the dC_D,T increment). |
| `PPO_Symmetric-stall.pdf` | The group's paper 1 — DP vs PPO on symmetric stall. |
| `Bunge.Pavone.Kroo.AIAAGNC18.pdf` | Bunge, Pavone, Kroo (AIAA GNC 2018). Source of the 0.33 s stall-detection latency used in `deteccion.py`. |
| `Barycentric_Interpolators.pdf` | Barycentric interpolation — the basis of `utils/utils.py`. |
| `Variable_Resolution_Discretization.pdf` | Variable-resolution discretization. |

A second directory, `papers/`, holds a hand-curated set that overlaps with this
one and adds Poliak's and Robbie's theses plus a 2014 paper on stall recovery
actions for single-engine light aeroplanes. It is gitignored under the same
policy.

Poliak's thesis is not loose bibliography: the code cites it. Both
`aircraft/grumman.py` and the CUDA kernel document that its eq. (12) writes the
transfer of moments to the CG with the opposite sign — giving nose-down where
nose-up belongs — and that this repo uses the standard derivation instead.

## Open item

Riley's `tau_e` (the engine time constant) is **not in `Riley.pdf`**: it appears
with a figure neither in Appendix A nor in the symbol list. It would have to be
tracked down in the report's references.
