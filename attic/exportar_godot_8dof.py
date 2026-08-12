"""Exporta una maniobra de 8-DOF a JSON para animarla en Godot.

Mismo esquema que exportar_blender.py del paper 1, pero con la actitud
COMPLETA: en un spin phi y psi no son cero, y la posicion lateral tampoco.

Dos cosas que hay que hacer bien:

  1. LA ACTITUD NO ES LA TRAYECTORIA. En 4-DOF alcanzaba con theta = gamma +
     alpha. Aca hace falta la matriz entera. Se arma la actitud inicial desde
     (chi=0, gamma, mu, alpha, beta) y de ahi en mas se INTEGRAN los angulos de
     Euler con las velocidades angulares del cuerpo, que es lo que el modelo
     realmente entrega:

         phi_dot   = p + (q sin(phi) + r cos(phi)) tan(theta)
         theta_dot = q cos(phi) - r sin(phi)
         psi_dot   = (q sin(phi) + r cos(phi)) / cos(theta)

     Asi la animacion muestra el avion girando de verdad, no un avion nivelado
     que baja.

  2. LOS EJES. Aeronautica usa x adelante, y ala derecha, z HACIA ABAJO.
     Godot/Blender son Z-up: hay cambio de mano. El JSON sale con la
     convencion ya resuelta (z hacia arriba, altitud positiva).

La posicion sale de integrar la velocidad: se arma (u,v,w) en cuerpo desde
V, alpha y beta, se rota a tierra con la actitud, y se integra.

Uso:
    .venv/bin/python exportar_godot_8dof.py <trayectoria.csv> <salida.json> \
        [--fps 60] [--z0 300] [--nombre spin_rescate]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def leer(path):
    with open(path) as f:
        r = csv.reader(f); cols = next(r); rows = [x for x in r]
    i = {c: k for k, c in enumerate(cols)}
    return {c: np.array([float(x[i[c]]) for x in rows]) for c in cols}


def dcm_desde_flujo(chi, gamma, mu, alpha, beta):
    """R_tierra<-cuerpo a partir de los angulos de trayectoria y de flujo."""
    cx, sx = np.cos(chi), np.sin(chi)
    cg, sg = np.cos(gamma), np.sin(gamma)
    cm, sm = np.cos(mu), np.sin(mu)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    R_ew = (np.array([[cx, -sx, 0.0], [sx, cx, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cm, -sm], [0.0, sm, cm]]))
    R_bw = np.array([[ca * cb, -ca * sb, -sa],
                     [sb,       cb,       0.0],
                     [sa * cb, -sa * sb,  ca]])
    return R_ew @ R_bw.T


def euler_desde_dcm(R_eb):
    """(phi, theta, psi) de R_cuerpo<-tierra = R_eb^T, convencion Rx(phi)Ry(theta)Rz(psi)."""
    R = R_eb.T
    theta = -np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    phi = np.arctan2(R[1, 2], R[2, 2])
    psi = np.arctan2(R[0, 1], R[0, 0])
    return phi, theta, psi


def exportar(csvp, out, fps=60, z0=300.0, nombre=None, v_stall=32.19):
    d = leer(csvp)
    t = d["t"]; dt = float(np.median(np.diff(t)))
    g, vn, al = d["gamma"], d["v_norm"], d["alpha"]
    be, mu = d["beta"], d["mu"]
    p, q, r = d["p"], d["q"], d["r"]

    # actitud inicial: desde los angulos de flujo, con rumbo chi = 0
    R0 = dcm_desde_flujo(0.0, g[0], mu[0], al[0], be[0])
    phi, th, psi = euler_desde_dcm(R0)

    n = len(t)
    E = np.zeros((n, 3)); P = np.zeros((n, 3))     # euler y posicion (NED: x norte, y este, z ABAJO)
    E[0] = (phi, th, psi)
    for k in range(n - 1):
        ph, te, ps = E[k]
        cph, sph, tte = np.cos(ph), np.sin(ph), np.tan(np.clip(te, -1.55, 1.55))
        cte = max(np.cos(te), 1e-6)
        dph = p[k] + (q[k] * sph + r[k] * cph) * tte
        dte = q[k] * cph - r[k] * sph
        dps = (q[k] * sph + r[k] * cph) / cte
        E[k + 1] = (ph + dph * dt, te + dte * dt, ps + dps * dt)
        # posicion: (u,v,w) en cuerpo -> tierra
        V = vn[k] * v_stall
        uvw = V * np.array([np.cos(al[k]) * np.cos(be[k]), np.sin(be[k]),
                            np.sin(al[k]) * np.cos(be[k])])
        cph2, sph2 = np.cos(ph), np.sin(ph)
        cte2, ste2 = np.cos(te), np.sin(te)
        cps, sps = np.cos(ps), np.sin(ps)
        R_be = np.array([
            [cte2 * cps, cte2 * sps, -ste2],
            [sph2 * ste2 * cps - cph2 * sps, sph2 * ste2 * sps + cph2 * cps, sph2 * cte2],
            [cph2 * ste2 * cps + sph2 * sps, cph2 * ste2 * sps - sph2 * cps, cph2 * cte2]])
        P[k + 1] = P[k] + (R_be.T @ uvw) * dt

    tv = np.arange(int(np.floor(t[-1] * fps)) + 1) / fps
    ip = lambda y: np.interp(tv, t, y)
    R2D = np.rad2deg
    frames = []
    for i, tt in enumerate(tv):
        frames.append({
            "frame": i, "t": round(float(tt), 5),
            # Godot/Blender Z-up: X = norte, Y = este, Z = altitud.
            #
            # El este NO se niega. vuelo.gd ya resuelve el cambio de mano por
            # su lado, en _base_actitud, que arma la actitud con
            # Basis(UP, -psi). Negar el este aca ADEMAS aplicaba la correccion
            # dos veces: la trompa se separaba de la velocidad como 2*psi y el
            # avion terminaba volando de cola. Con psi = 0 --las 15 maniobras
            # del paper 1-- el error es invisible, y por eso sobrevivio.
            # Medido: con el este negado el angulo trompa-velocidad se desvia
            # 59 deg de alpha (max 145); sin negar, 0.2 deg (max 2.6).
            "x": round(float(ip(P[:, 0])[i]), 4),
            "y": round(float(ip(P[:, 1])[i]), 4),
            "z": round(float(z0 - ip(P[:, 2])[i]), 4),
            "phi_deg": round(float(R2D(ip(E[:, 0])[i])), 4),
            "theta_deg": round(float(R2D(ip(E[:, 1])[i])), 4),
            "psi_deg": round(float(R2D(ip(E[:, 2])[i])), 4),
            "elevator_deg": round(float(R2D(ip(d["de"])[i])), 3),
            "aileron_deg": round(float(R2D(ip(d["da"])[i])), 3),
            "rudder_deg": round(float(R2D(ip(d["dr"])[i])), 3),
            "throttle": round(float(ip(d["dt_ctrl"])[i]), 4),
            "gamma_deg": round(float(R2D(ip(g)[i])), 3),
            "alpha_deg": round(float(R2D(ip(al)[i])), 3),
            "beta_deg": round(float(R2D(ip(be)[i])), 3),
            "mu_deg": round(float(R2D(ip(mu)[i])), 3),
            "p_deg_s": round(float(R2D(ip(p)[i])), 3),
            "r_deg_s": round(float(R2D(ip(r)[i])), 3),
            "v_over_vs": round(float(ip(vn)[i]), 4),
            "control": int(round(float(ip(d["quien"])[i]))) if "quien" in d else 1,
        })
    # vuelo.gd exige estos campos y chequea entry.gamma0_deg contra la
    # entrada declarada en ENTRADAS: renombrar un archivo no puede hacer que
    # el HUD diga una cosa y se vuele otra.
    doc = {
        "arm": nombre or Path(csvp).stem,
        "brazo": nombre or Path(csvp).stem,
        "fps": fps, "n_frames": len(frames),
        "duration_s": round(float(t[-1]), 3),
        "altitude_lost_m": round(float(d["h"][-1]), 3),
        "recovery_t_s": round(float(t[-1]), 3),
        "recovery_frame": len(frames) - 1,
        "tail_s": 0.0,
        "entry": {"alpha0_deg": round(float(np.rad2deg(al[0])), 2),
                  "v0_over_vs": round(float(vn[0]), 4),
                  "gamma0_deg": round(float(np.rad2deg(g[0])), 2),
                  "q0_deg_s": round(float(np.rad2deg(q[0])), 2),
                  "mu0_deg": round(float(np.rad2deg(mu[0])), 2),
                  "beta0_deg": round(float(np.rad2deg(be[0])), 2)},
        "aircraft": "Grumman American AA-1 Yankee (Riley, NASA TM-86309)",
        "convention": ("Godot/Blender Z-up. X = norte, Y = este, Z = altitud. "
                       "Actitud COMPLETA (phi, theta, psi) integrada de las "
                       "velocidades angulares del cuerpo; NO es la trayectoria. "
                       "control: 0 = mandos fijos pro-spin, 1 = politica."),
        "frames": frames,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print("escrito %s  (%d frames, %.2f s, %.1f m)"
          % (out, len(frames), t[-1], d["h"][-1]))
    print("  phi  %+7.1f .. %+7.1f deg   (banqueo)" % (R2D(E[:,0]).min(), R2D(E[:,0]).max()))
    print("  theta %+6.1f .. %+7.1f deg   (cabeceo)" % (R2D(E[:,1]).min(), R2D(E[:,1]).max()))
    print("  psi  %+7.1f .. %+7.1f deg   (rumbo, %.1f vueltas)"
          % (R2D(E[:,2]).min(), R2D(E[:,2]).max(), (R2D(E[:,2]).max()-R2D(E[:,2]).min())/360))
    print("  recorrido: x %.0f m, y %.0f m, altura %.0f m" % (P[-1,0], P[-1,1], -P[-1,2]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--z0", type=float, default=300.0)
    ap.add_argument("--nombre", default=None)
    a = ap.parse_args()
    exportar(a.csv, a.out, a.fps, a.z0, a.nombre)
