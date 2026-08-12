# refs — bibliografía

Los PDFs están en disco pero **fuera de git** (tienen copyright; ver
`.gitignore`). Esta lista es el registro de qué hay.

| archivo | referencia |
|---|---|
| `Riley.pdf` | Riley, D. R. (1985). Tablas aerodinámicas del Grumman AA-1. **La fuente del modelo**: Tabla I (masa, inercias, geometría), Tabla III (coeficientes vs α para C_T = 0 y 0.5), Apéndice A ec. (A4) (retardo de primer orden del motor), Apéndice B (incremento dC_D,T). |
| `PPO_Symmetric-stall.pdf` | El paper 1 del grupo — DP vs PPO en pérdida simétrica. |
| `Bunge.Pavone.Kroo.AIAAGNC18.pdf` | Bunge, Pavone, Kroo (AIAA GNC 2018). |
| `Barycentric_Interpolators.pdf` | Interpolación baricéntrica — la base de `utils/utils.py`. |
| `Variable_Resolution_Discretization.pdf` | Discretización de resolución variable. |

## Pendiente

El valor de `tau_e` (constante de tiempo del motor) **no está en `Riley.pdf`**:
ni en el Apéndice A ni en la lista de símbolos aparece con cifra. Hay que
buscarlo en las referencias del informe.
