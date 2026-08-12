# Resultados 4-DOF — 2026-08-11

Todo esto se corrio en la **3090** (`nromero@100.68.2.122`), rama
`4dof-riley-thrust`, y **todo con `THRUST_MODEL=riley`**. El default del codigo
es `paper1`: si corres algo sin esa variable, los numeros no van a coincidir
(la canonica da -13.3 m en vez de -6.8).

## Las dos politicas

| archivo | grilla | estados | corrida |
|---|---|---|---|
| `politicas/SymmetricStall_alpha_m10_40_FILLED.npz` | 56x81x80x41 | 14.878.080 | 47 iter, **4 h 18** |
| `politicas/SymmetricStall_alpha_m10_40_RAW.npz` | idem, **sin rellenar** | | (la misma, antes del fill) |
| `politicas/SymmetricStall_policy.npz` | 56x41x60x41 | 5.648.160 | 23 iter, **43 min** |

**La primera** es la grilla nueva: `alpha` recolocado a **[-10, +40]** (el dominio
exacto que Riley tabula) y `V` bajado a **[0.4, 2.0]**. La segunda es la **grilla
del paper 1** con el empuje de Riley — se corrio para separar el efecto de la
grilla del efecto del empuje, y **todavia no se analizo**.

Config identica a la del paper en las dos: `gamma=1.0`, `theta=5e-6`,
`maximum_iterations=20.000`, `n_micro=10`, dt fijo, todo el shaping en cero.

## Las carpetas

    0_heatmaps/        los heatmaps y la trayectoria que genera main.py solo
    1_canonica_fill/   80 archivos: comparacion_fill_vXXX.png (16 velocidades,
                       rellenada vs sin rellenar) + las individuales + PDF
    2_barrido_IC/      barrido_v0.png (la curva h_min(V0)) y las dos familias
                       de trayectorias superpuestas
    3_maniobras/       lo de CAA/FAA y las ablaciones  <- LO QUE MAS IMPORTA
    politicas/         los .npz

## Los numeros

### Trayectoria canonica (gamma0=0, alpha0=20, q0=0)

| V0/Vs | h_min rellenada | h_min sin rellenar | el fill vale |
|---|---|---|---|
| 0.80 | -16.859 | -16.963 | 10.4 cm |
| 0.85 | **-9.344** | -9.589 | 24.5 cm |
| 0.86 | -7.904 | -8.199 | 29.5 cm |
| 0.89 | -3.897 | -4.369 | 47.2 cm |
| 0.90 | -2.697 | -3.233 | 53.6 cm |
| 0.93 | -0.036 | -0.774 | 73.8 cm |
| 0.95 | **+0.000** | -0.390 | -- |

- **A 0.95 Vs no hay maniobra**: con Riley el avion no se hunde. Por eso la IC
  canonica baja a 0.85/0.86.
- La perdida es casi lineal entre 0.80 y 0.90: **1.42 m por centesima de Vs**.
- **`h_min` es la metrica robusta**, no el `dh` final: no depende del criterio
  de corte. El umbral `has_dived` se bajo de -2 a **-1 grado** porque a 0.90 la
  politica rellenada solo llega a -1.82 y no disparaba el corte.

### Invariantes (valen en las 16 entradas)

- `delta_t` medio **1.000 exacto** en la politica rellenada, siempre.
- `alpha` final entre **13.66 y 14.11 grados**: la maniobra termina pegada a la
  frontera de perdida. Es el modo deslizante del paper 1.
- `alpha_min` varia solo **0.26 grados** entre 0.85 y 0.90 mientras la perdida
  cambia por un factor 3.5: **la maniobra es la misma, cambia cuanto dura**.

### Maniobras CAA / FAA (`3_maniobras/maniobras_v086.png`), a 0.86 Vs

| | dh | alpha_max | estado |
|---|---|---|---|
| optimo | -7.904 | 20.00 | recuperado |
| CAA alpha-hold | -21.850 | 20.07 | recuperado |
| FAA alpha-hold | -26.055 | 20.07 | recuperado |
| CAA full-pull | -96.321 | **35.82** | TIMEOUT |
| FAA full-pull | -99.156 | 35.24 | TIMEOUT |

El `full-pull` **re-estanca el ala** y no recupera: sostiene la nota al pie de
la tabla del paper con un numero.

### Ablacion: gatillo contra potencia (`3_maniobras/ablacion_v086.png`)

De los 13.95 m que pierde el CAA contra el optimo:

    la rampa de potencia de 2 s explica    12.90 m   (92.5 %)
    el gatillo tardio del elevador           0.59 m   ( 4.2 %)

`gatillo 14 + escalon` da **-8.948 m**, a un metro del optimo: el procedimiento,
**sin tocarle nada al elevador** y solo metiendo la potencia de golpe, casi
alcanza al optimo.

### Solo potencia, mismo elevador (`3_maniobras/potencia_pura_v089.png`), 0.89 Vs

El `delta_e` sale de la politica en los tres brazos; lo unico que cambia es la
rampa de potencia:

| | h_min | penalizacion |
|---|---|---|
| DP, rampa 0.6 s | -7.907 | -- |
| CAA, rampa 2 s desde t=0 | -15.857 | -7.95 m |
| FAA, rampa 2 s tras el nose-down | -20.050 | -12.14 m |

Los **0.38 s** que FAA espera al nose-down cuestan **4.19 m**, o sea
**11 metros por segundo de demora en aplicar potencia**.

### Dinamica de motor de Riley (`3_maniobras/potencia_riley_v089_tau050.png`)

Riley, Apendice A ec. (A4), incorpora un **retardo de primer orden** entre la
palanca y el empuje: `delta_t = 1/(tau_e s + 1) delta_t,c`. **El modelo NO lo
tiene** (el throttle entra directo a `_compute_ct`, y no hay estado para el),
y **Riley no publica el valor de `tau_e`**.

| tau_e | DP | CAA | FAA | CAA-DP | FAA-CAA |
|---|---|---|---|---|---|
| 0.00 | -7.907 | -15.857 | -20.050 | -7.95 | -4.19 |
| 0.25 | -10.747 | -18.501 | -22.612 | -7.75 | -4.11 |
| 0.50 | -13.498 | -21.099 | -25.136 | -7.60 | -4.04 |
| 1.00 | -18.495 | -25.788 | -29.644 | -7.29 | -3.86 |

**La perdida absoluta depende fuerte de `tau_e`; la comparacion entre brazos casi
nada (8 %).** O sea se puede reportar "esperar al nose-down cuesta 4 m" sin
comprometerse con un valor de `tau_e`; lo que NO se puede reportar sin fijarlo
es la perdida absoluta del optimo.

### Piloto con retardo (`3_maniobras/piloto_realista_v086.png`)

Con **1 s de retardo humano** siguiendo el `delta_e` del DP, **choca siempre**,
incluso con motor ideal: mantiene la picada un segundo de mas, `alpha` cae a
**-19 grados**, `gamma` a -30, despues sobre-corrige con `q = +95 deg/s` y
re-estanca. Es una **PIO**.

Conclusion: **la politica optima es una cota, no un procedimiento volable.** Su
pulso de elevador dura 0.2 s y no tolera retardo humano. Los procedimientos
CAA/FAA son robustos justamente porque usan gatillos de estado (`alpha < 14`) y
no un perfil temporal.

## Como reproducir

    cd ~/stall-spin-recovery-dp
    THRUST_MODEL=riley .venv/bin/python <script> results/politicas/<npz> <args>

| script | que hace |
|---|---|
| `canonica_filled_vs_raw.py <raw> <filled> V0...` | trayectorias rellenada vs cruda |
| `barrido_v0.py <raw> <filled>` | la curva `h_min(V0)` |
| `familia2.py <npz> <salida.png> V0...` | trayectorias superpuestas |
| `maniobras_086.py <npz> <V0> [alpha0]` | optimo vs CAA vs FAA |
| `ablacion.py <npz> <V0>` | factorial gatillo x potencia |
| `potencia_pura.py <npz> <V0>` | mismo elevador, distinta rampa |
| `potencia_riley.py <npz> <V0> <tau_fig> <taus...>` | con el motor de Riley |
| `piloto_realista.py <npz> <V0> <tau_h> <taus_m...>` | retardo humano + motor |
| `fill_terminal_policy.py <npz>` | el relleno terminal (sobreescribe, deja `.npz.raw`) |

## Pendientes

1. **Analizar la corrida de la grilla del paper** (`SymmetricStall_policy.npz`),
   que es la que separa el efecto de la grilla del efecto del empuje.
2. **`main.py` no registra `THRUST_MODEL` ni el CG en el log ni en el `.npz`.**
   Hoy no hay forma de saber, mirando una politica, con que modelo se entreno.
3. El valor de `tau_e` de Riley: no esta en el Apendice A ni en la lista de
   simbolos con cifra. Habria que buscarlo en las referencias del informe.
4. La grilla del 6-DOF sigue con el problema de la caja no absorbente
   (el 4.99 % de sus celdas vale menos que estrellarse).
