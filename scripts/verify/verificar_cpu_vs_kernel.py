"""Comprueba que el modelo CPU y el kernel CUDA calculan la MISMA dinamica.

Las tablas aerodinamicas estan duplicadas -- en aircraft/grumman.py y en el
.cu embebido en PolicyIteration.py -- y las formulas tambien. Nada garantiza
que se mantengan sincronizadas: el kernel entrena la politica y el CPU la
evalua, asi que si divergen, la politica se optimiza para un avion levemente
distinto del que despues se simula.

Comparar las tablas no alcanza: dos tablas iguales con formulas distintas dan
resultados distintos. Esto compara las DERIVADAS evaluadas sobre estados
aleatorios de toda la grilla, que es donde se manifiesta cualquier diferencia.

Correr despues de tocar cualquier cosa aerodinamica:

    python verificar_cpu_vs_kernel.py

Sale 0 si coinciden a precision de float32, 1 si no.
"""
import logging
import os
import sys

import numpy as np

logging.disable(logging.INFO)

TOL_REL = 1e-4          # holgado para float32 acumulado en ~20 operaciones
N_ESTADOS = 4000
SEMILLA = 3

DEFINES = "\n".join(
    f"#define {k} {v}f" for k, v in [
        ("W_Q_PENALTY", "0.0"), ("W_ALPHA_BARRIER_POS", "0.0"),
        ("W_ALPHA_BARRIER_NEG", "0.0"), ("W_CRASH_PENALTY", "1000.0"),
        ("W_CONTROL_EFFORT", "0.0"), ("W_THROTTLE_BONUS", "0.0"),
        ("DXCG_OVER_CHORD", "0.0"),
    ])
# THRUST_RILEY no lleva sufijo f: es un flag entero para el preprocesador. Y
# tiene que salir de la MISMA variable de entorno que el modelo CPU, o este
# script compara el kernel en un modo contra el CPU en el otro -- que fue
# exactamente lo que reporto la primera vez que se corrio con riley.
DEFINES += f"\n#define THRUST_RILEY {1 if os.environ.get('THRUST_MODEL','paper1').lower()=='riley' else 0}\n"

# El CG sale de las MISMAS variables de entorno que la planta. Fijarlo en cero
# aca compararia el kernel con el CG de Riley contra un CPU con el CG corrido
# -- el mismo error que ya se cometio una vez con THRUST_MODEL.
DEFINES += (
    "\n#define CG_AFT_M " + repr(float(os.environ.get("CG_AFT_M", 0.0))) + "f"
    + "\n#define CG_RIGHT_M " + repr(float(os.environ.get("CG_RIGHT_M", 0.0))) + "f"
    + "\n#define CG_BELOW_M " + repr(float(os.environ.get("CG_BELOW_M", 0.0))) + "f\n"
)

WRAPPER = r'''
extern "C" __global__ void test_derivs(
        const float* st, const float* ac, int n,
        float v_stall, float k_thrust, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dg, dv, da, dq;
    derivs_4dof(st[i*4+0], st[i*4+1], st[i*4+2], st[i*4+3],
                ac[i*2+0], ac[i*2+1], v_stall, k_thrust, dg, dv, da, dq);
    out[i*4+0] = dg; out[i*4+1] = dv; out[i*4+2] = da; out[i*4+3] = dq;
}
'''


def cuda_body():
    txt = open("PolicyIteration.py").read()
    marca = "cuda_source = reward_defines + r'''"
    i = txt.index(marca) + len(marca)
    return txt[i:txt.index("'''", i)]


def derivs_cpu(ap, g, v, a, q, de, th, vs):
    """Llama a LA PLANTA. Antes esto era una copia a mano de sus expresiones,
    y una copia a mano solo verifica que el kernel coincida con la copia."""
    return ap.derivatives(g, v, a, q, de, th)


def main():
    import cupy as cp
    from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

    kern = cp.RawModule(code=DEFINES + cuda_body() + WRAPPER,
                        options=('-std=c++11',)).get_function("test_derivs")
    ap = SymmetricFullGrumman()
    vs, kt = ap.STALL_AIRSPEED, ap.THROTTLE_LINEAR_MAPPING

    rng = np.random.default_rng(SEMILLA)
    n = N_ESTADOS
    # el piso de velocidad acompana al de la grilla: por debajo de 0.785 Vs se
    # activa dCD_T, y si el muestreo no baja hasta ahi el termino no se prueba
    S = np.stack([rng.uniform(-1.5, 0.08, n), rng.uniform(0.4, 2.0, n),
                  rng.uniform(-0.69, 0.34, n),
                  rng.uniform(-0.87, 0.87, n)], 1).astype(np.float32)
    Ac = np.stack([rng.uniform(np.deg2rad(-25), np.deg2rad(15), n),
                   rng.uniform(0, 1, n)], 1).astype(np.float32)

    out = cp.zeros((n, 4), dtype=cp.float32)
    kern((n // 256 + 1,), (256,),
         (cp.asarray(S), cp.asarray(Ac), np.int32(n),
          np.float32(vs), np.float32(kt), out))
    gpu = cp.asnumpy(out)

    cpu = np.array([derivs_cpu(ap, *map(float, S[i]), *map(float, Ac[i]), vs)
                    for i in range(n)])

    rel = np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 1e-6)
    # el veredicto sale del MAXIMO, no de la mediana: un termino que solo
    # aparece en una esquina del dominio -- una tabla mal transcripta cerca de
    # 40 grados, dCD_T por debajo de 0.785 Vs -- no mueve la mediana de 4000
    # muestras, y era exactamente lo que habia que detectar
    peor = 0.0
    print(f"{n} estados aleatorios, CPU vs kernel CUDA:")
    for i, nom in enumerate(["gamma_dot", "v_dot", "alpha_dot", "q_dot"]):
        med = float(np.median(rel[:, i]))
        mx = float(rel[:, i].max())
        p99 = float(np.percentile(rel[:, i], 99))
        peor = max(peor, mx)
        j = int(rel[:, i].argmax())
        print(f"  {nom:10s} mediana {med:.2e}   p99 {p99:.2e}   MAX {mx:.2e}"
              f"   (peor caso: V={S[j,1]:.2f} Vs, alpha={np.rad2deg(S[j,2]):+.1f} deg,"
              f" dt={Ac[j,1]:.2f})")
    # v_dot cruza cero (la resistencia neta cambia de signo con C_T), asi que
    # ahi el error relativo se dispara sin que nada este mal. El veredicto usa
    # un criterio mixto: relativo donde la derivada tiene magnitud, absoluto
    # donde se cancela. La escala es la desviacion tipica de cada canal.
    escala = np.maximum(cpu.std(axis=0), 1e-9)
    mixto = np.abs(gpu - cpu) / np.maximum(np.abs(cpu), 0.01 * escala)
    print()
    for i, nom in enumerate(["gamma_dot", "v_dot", "alpha_dot", "q_dot"]):
        j = int(mixto[:, i].argmax())
        print(f"  {nom:10s} escala {escala[i]:.3e}   abs MAX {np.abs(gpu-cpu)[:, i].max():.3e}"
              f"   mixto MAX {mixto[:, i].max():.2e}"
              f"   (cpu={cpu[j, i]:+.3e} gpu={gpu[j, i]:+.3e})")
    peor = float(mixto.max())
    ok = peor < TOL_REL
    print(f"\n{'COINCIDEN' if ok else 'DIFIEREN'} "
          f"(peor MAXIMO {peor:.2e}, tolerancia {TOL_REL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
