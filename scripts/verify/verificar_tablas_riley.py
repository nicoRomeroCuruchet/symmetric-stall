"""Compara las tablas del codigo contra la Tabla III del PDF de Riley.

Hasta ahora se verificaba que el kernel CUDA coincidiera con el modelo de
Python. Eso no dice nada sobre si alguno de los dos dice lo que dice el
informe: las dos copias pueden coincidir y estar las dos mal, que es
exactamente lo que pasaba.

Esto lee papers/riley.pdf, extrae la Tabla III y compara valor por valor
contra los arreglos del codigo. El texto sale de un escaneo de 1985 y el OCR
confunde caracteres de forma sistematica -- comas por puntos, O por cero, Z
por dos, q por cuatro -- asi que se normaliza antes de comparar y todo lo que
no se puede leer se reporta como ILEGIBLE en vez de adivinarse.

Usage:  PYTHONPATH=. python verificar_tablas_riley.py
"""
import re
import subprocess
import sys

import numpy as np

from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

PDF = "papers/riley.pdf"
GRADO_A_RAD = 57.2958  # el codigo guarda las derivadas de control por radian

# Bloque de la Tabla III -> (primera linea, ultima linea) en el volcado de
# pdftotext -layout, y que columna es cada grupo de tres.
BLOQUES = [
    # (linea inicio, linea fin, [(nombre_grupo, atributo_ct0, atributo_ct05)])
    (1424, 1441, [("CL_o", "_CL_O_TABLE", "_CL_O_TABLE_CT05"),
                  ("CL_de", "_CL_DE_TABLE_CT0", "_CL_DE_TABLE_CT05"),
                  ("CL_df", None, None)]),
    (1445, 1462, [("dCL_beta", None, None),
                  ("CL_q", "_CL_Q_TABLE", "_CL_Q_TABLE_CT05"),
                  ("CL_adot", "_CL_ADOT_TABLE_CT0", "_CL_ADOT_TABLE_CT05")]),
    (1472, 1489, [("CD_o", "_CD_O_TABLE", "_CD_O_TABLE_CT05"),
                  ("CD_de", "_CD_DE_TABLE_CT0", "_CD_DE_TABLE_CT05"),
                  ("CD_de2", "_CD_DE2_TABLE_CT0", "_CD_DE2_TABLE_CT05")]),
    (1517, 1534, [("Cm_o", "_CM_O_TABLE", "_CM_O_TABLE_CT05"),
                  ("Cm_de", "_CM_DE_TABLE_CT0", "_CM_DE_TABLE_CT05"),
                  ("Cm_df", None, None)]),
    (1538, 1556, [("dCm_beta", None, None),
                  ("Cm_q", "_CM_Q_TABLE", "_CM_Q_TABLE_CT05"),
                  ("Cm_adot", "_CM_ADOT_TABLE_CT0", "_CM_ADOT_TABLE_CT05")]),
]

# Las derivadas de control estan tabuladas POR GRADO y el codigo las guarda
# por radian; las de q y alpha-punto multiplican una cantidad ya adimensional
# y van tal cual. CD_de2 lleva ademas el 1e-2 del encabezado de su columna.
ESCALA = {
    "_CL_DE_TABLE_CT0": GRADO_A_RAD, "_CL_DE_TABLE_CT05": GRADO_A_RAD,
    "_CM_DE_TABLE_CT0": GRADO_A_RAD, "_CM_DE_TABLE_CT05": GRADO_A_RAD,
    "_CD_DE_TABLE_CT0": GRADO_A_RAD, "_CD_DE_TABLE_CT05": GRADO_A_RAD,
    "_CD_DE2_TABLE_CT0": 1e-2 * GRADO_A_RAD ** 2,
    "_CD_DE2_TABLE_CT05": 1e-2 * GRADO_A_RAD ** 2,
}


# ---------------------------------------------------------------------------
# Tablas LATERALES, III(d) fuerza lateral, III(e) guiniada, III(f) alabeo.
# Solo existen en las ramas de 6 y 8 DOF; en el 4-DOF se saltean.
#
# La forma se repite en las tres: dos bloques de tres grupos. En el primer
# bloque el tercer grupo (la derivada de aleron) NO tiene desdoble por C_T y
# ocupa una sola columna, asi que esa fila trae 8 tokens y no 9.
BLOQUES_LAT = [
    (1565, 1580, 8, [("CY_o", "_CY_O_TABLE", "_CY_O_TABLE_CT05"),
                     ("CY_beta", "_CY_BETA_TABLE_CT0", "_CY_BETA_TABLE_CT05"),
                     ("CY_da", "_CY_DA_TABLE", None)]),
    (1583, 1603, 9, [("CY_dr", "_CY_DR_TABLE_CT0", "_CY_DR_TABLE_CT05"),
                     ("CY_r", "_CY_RHAT_TABLE_CT0", "_CY_RHAT_TABLE_CT05"),
                     ("CY_p", "_CY_PHAT_TABLE", None)]),
    (1614, 1629, 8, [("Cn_o", "_CN_O_TABLE", "_CN_O_TABLE_CT05"),
                     ("Cn_beta", "_CN_BETA_TABLE_CT0", "_CN_BETA_TABLE_CT05"),
                     ("Cn_da", "_CN_DA_TABLE", None)]),
    (1633, 1649, 9, [("Cn_dr", "_CN_DR_TABLE_CT0", "_CN_DR_TABLE_CT05"),
                     ("Cn_r", "_CN_RHAT_TABLE_CT0", "_CN_RHAT_TABLE_CT05"),
                     ("Cn_p", "_CN_PHAT_TABLE_CT0", "_CN_PHAT_TABLE_CT05")]),
    (1659, 1674, 8, [("Cl_o", "_CL_ROLL_O_TABLE", "_CL_ROLL_O_TABLE_CT05"),
                     ("Cl_beta", "_CL_ROLL_BETA_TABLE_CT0", "_CL_ROLL_BETA_TABLE_CT05"),
                     ("Cl_da", "_CL_ROLL_DA_TABLE", None)]),
    (1678, 1694, 9, [("Cl_dr", "_CL_ROLL_DR_TABLE", None),
                     ("Cl_r", "_CL_ROLL_RHAT_TABLE_CT0", "_CL_ROLL_RHAT_TABLE_CT05"),
                     ("Cl_p", "_CL_ROLL_PHAT_TABLE_CT0", "_CL_ROLL_PHAT_TABLE_CT05")]),
]

# Las derivadas de CONTROL y de BETA estan tabuladas por grado; las de p y r
# multiplican una cantidad ya adimensional y van tal cual.
for _n in ("_CY_BETA_TABLE_CT0", "_CY_BETA_TABLE_CT05", "_CY_DA_TABLE",
           "_CY_DR_TABLE_CT0", "_CY_DR_TABLE_CT05",
           "_CN_BETA_TABLE_CT0", "_CN_BETA_TABLE_CT05", "_CN_DA_TABLE",
           "_CN_DR_TABLE_CT0", "_CN_DR_TABLE_CT05",
           "_CL_ROLL_BETA_TABLE_CT0", "_CL_ROLL_BETA_TABLE_CT05",
           "_CL_ROLL_DA_TABLE", "_CL_ROLL_DR_TABLE"):
    ESCALA[_n] = GRADO_A_RAD


# Entradas donde el OCR es irrecuperable y se resolvieron mirando la PAGINA
# del PDF como imagen. El valor es el que se lee ahi; el script las reporta
# como confirmadas en vez de como diferencias, con la pagina al lado para que
# cualquiera pueda repetir la lectura.
#
# El patron es siempre el mismo: la "q" del escaneo es un 9 (-,349q = -.3499,
# -*Olq3 = -.0193), y los ceros iniciales se comen el punto decimal.
CONFIRMADOS = {
    ("_CL_DE_TABLE_CT05", 1): (0.0134, "pag. 32, CL_de CT=0.5, alpha=-5"),
    ("_CD_O_TABLE_CT05", 1): (-0.3499, "pag. 33, CD_o CT=0.5, alpha=-5"),
    ("_CD_DE2_TABLE_CT05", 0): (0.0000, "pag. 33, CD_(de)2 CT=0.5, alpha=-10"),
    ("_CM_O_TABLE", 0): (0.2700, "pag. 34, Cm_o CT=0.0, alpha=-10"),
    ("_CM_DE_TABLE_CT0", 2): (-0.0193, "pag. 34, Cm_de CT=0.0, alpha=0"),
    ("_CM_DE_TABLE_CT05", 13): (-0.0153, "pag. 34, Cm_de CT=0.5, alpha=40"),
    ("_CM_Q_TABLE_CT05", 10): (-19.0600, "pag. 34, Cm_q CT=0.5, alpha=25"),
    ("_CM_ADOT_TABLE_CT05", 9): (0.0000, "pag. 34, Cm_adot CT=0.5, alpha=20"),
    # --- laterales, resueltas contra las paginas 35, 36 y 37 ---
    ("_CY_O_TABLE_CT05", 11): (-0.0540, "pag. 35, CY_o CT=0.5, alpha=30"),
    ("_CY_BETA_TABLE_CT0", 11): (-0.00600, "pag. 35, CY_beta CT=0.0, alpha=30"),
    ("_CY_BETA_TABLE_CT05", 11): (-0.02020, "pag. 35, CY_beta CT=0.5, alpha=30"),
    ("_CY_DA_TABLE", 4): (-0.000140, "pag. 35, CY_da, alpha=10"),
    ("_CY_DA_TABLE", 12): (0.0, "pag. 35, CY_da, alpha=35"),
    ("_CY_DA_TABLE", 13): (0.0, "pag. 35, CY_da, alpha=40"),
    ("_CN_BETA_TABLE_CT0", 0): (0.00250, "pag. 36, Cn_beta CT=0.0, alpha=-10"),
    ("_CN_DA_TABLE", 5): (0.0, "pag. 36, Cn_da, alpha=12"),
    ("_CN_DR_TABLE_CT05", 1): (-0.00299, "pag. 36, Cn_dr CT=0.5, alpha=-5"),
    ("_CN_DR_TABLE_CT05", 6): (-0.00369, "pag. 36, Cn_dr CT=0.5, alpha=14"),
    ("_CN_RHAT_TABLE_CT05", 0): (-0.2900, "pag. 36, Cn_r CT=0.5, alpha=-10"),
    ("_CN_RHAT_TABLE_CT05", 3): (-0.2900, "pag. 36, Cn_r CT=0.5, alpha=5"),
    ("_CN_PHAT_TABLE_CT0", 9): (0.0400, "pag. 36, Cn_p CT=0.0, alpha=20"),
    ("_CL_ROLL_BETA_TABLE_CT0", 12): (-0.00400, "pag. 37, Cl_beta CT=0.0, alpha=35"),
    ("_CL_ROLL_DR_TABLE", 13): (0.0, "pag. 37, Cl_dr, alpha=40"),
    ("_CL_ROLL_PHAT_TABLE_CT0", 6): (-0.2200, "pag. 37, Cl_p CT=0.0, alpha=14"),
    ("_CY_DR_TABLE_CT0", 3): (0.00295, "pag. 35, CY_dr CT=0.0, alpha=5"),
    ("_CY_DR_TABLE_CT05", 0): (0.00589, "pag. 35, CY_dr CT=0.5, alpha=-10"),
    ("_CY_RHAT_TABLE_CT0", 10): (-0.2500, "pag. 35, CY_r CT=0.0, alpha=25"),
    ("_CY_RHAT_TABLE_CT0", 13): (0.0, "pag. 35, CY_r CT=0.0, alpha=40"),
    ("_CY_PHAT_TABLE", 7): (0.0380, "pag. 35, CY_p, alpha=16"),
    ("_CY_PHAT_TABLE", 13): (0.0, "pag. 35, CY_p, alpha=40"),
}

# confusiones sistematicas del OCR sobre digitos
OCR = str.maketrans({"O": "0", "o": "0", "C": "0", "c": "0", "Q": "0",
                     "l": "1", "I": "1", "i": "1", "|": "1",
                     "Z": "2", "z": "2", "S": "5", "s": "5", "§": "5",
                     "q": "4", "B": "8", "b": "6", "G": "6",
                     ",": ".", "*": ".", "°": ".", "•": "."})


def leer_pdf():
    r = subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "92", PDF, "-"],
                       capture_output=True, text=True)
    if r.returncode != 0 or len(r.stdout) < 10000:
        raise SystemExit(
            f"NO SE PUDO LEER {PDF}. Sin el informe no hay nada contra que "
            f"comparar, y salir en verde seria peor que fallar.")
    return r.stdout.split("\n")


def numeros(linea):
    """Devuelve los tokens numericos de una fila, o None donde no se puede leer.

    El punto decimal inicial es lo que mas se degrada: ".0051" sale como
    "o0051", "*0350" o ",0062". Se normaliza aparte, porque mapearlo como
    digito convierte 0.0051 en 51 y eso parece un error de transcripcion
    cuando es un error de lectura.
    """
    # la columna CD(de)2 lleva el "x10-2" del encabezado repetido en la fila de
    # alpha=-10 y pegado a los valores, que ademas quedan sin separador
    linea = re.sub(r"x[lI1][oO0]-?2?", " ", linea)
    fuera = []
    for tok in linea.split():
        # el escaneo deja marcas de pagina sueltas ("I", "0") al margen; si se
        # las toma por numeros corren TODAS las columnas de esa fila un lugar.
        # Pero un valor ilegible tambien puede quedar sin ningun digito
        # ("-.nl_"), y ese SI ocupa columna: se distingue por largo.
        if not any(ch.isdigit() for ch in tok):
            if len(tok) >= 3 and any(c in "._" for c in tok):
                fuera.append(None)
            continue
        signo = ""
        cuerpo = tok
        if cuerpo[:1] in "-+":
            signo, cuerpo = cuerpo[0], cuerpo[1:]
        if cuerpo[:1] in "oO0Cc*,.°•" and not cuerpo.startswith("0."):
            cuerpo = "." + cuerpo[1:]
        t = (signo + cuerpo).translate(OCR)
        # un token legible es signo opcional, digitos y a lo sumo un punto
        if re.fullmatch(r"[-+]?\d*\.?\d+", t) and any(ch.isdigit() for ch in t):
            fuera.append(float(t))
        elif re.search(r"[\d._]", tok):
            fuera.append(None)          # habia algo numerico pero ilegible
    return fuera


def main():
    lineas = leer_pdf()
    a = SymmetricFullGrumman()
    total = fallos = ilegibles = confirmados = 0
    print("Tabla III del PDF contra los arreglos del codigo\n")

    for ini, fin, ncol, grupos in [(a_, b_, 9, c_) for a_, b_, c_ in BLOQUES] + BLOQUES_LAT:
        filas = []
        for l in lineas[ini - 1:fin]:
            if not l.strip():
                continue
            v = numeros(l)
            if len(v) >= ncol:
                filas.append(v[:ncol])
        if len(filas) != 14:
            print(f"  !! bloque {ini}-{fin}: lei {len(filas)} filas, esperaba 14")
            continue

        M = np.array([[np.nan if x is None else x for x in f] for f in filas])
        # con 8 columnas el ultimo grupo trae una sola, asi que el indice del
        # tercer grupo no es 3*gi sino el que quede
        for gi, (nombre, at0, at05) in enumerate(grupos):
            base = 3 * gi + 1
            for col, attr in ((base, at0), (base + 1, at05)):
                if attr is None or not hasattr(a, attr) or col >= M.shape[1]:
                    continue
                pdf = M[:, col]
                cod = np.asarray(getattr(a, attr), dtype=np.float64)
                esc = ESCALA.get(attr, 1.0)
                pdf_esc = pdf * esc
                malas = np.isnan(pdf)
                dif = np.abs(pdf_esc - cod) > np.maximum(1e-4 * np.abs(cod), 1e-7)
                dif &= ~malas
                total += 14
                for i in np.where(malas | dif)[0]:
                    ref = CONFIRMADOS.get((attr, i))
                    if ref is not None and abs(ref[0] * esc - cod[i]) <= max(
                            1e-6 * abs(cod[i]), 1e-9):
                        confirmados += 1
                        print(f"  {nombre:<10} alpha[{i:2d}]  OCR ilegible, "
                              f"confirmado contra la imagen: {ref[0]:+.6g}"
                              f"  ({ref[1]})")
                        continue
                    if malas[i]:
                        ilegibles += 1
                        print(f"  {nombre:<10} alpha[{i:2d}]  ILEGIBLE y SIN "
                              f"confirmar, codigo {cod[i]:+.6g}")
                    else:
                        fallos += 1
                        print(f"  {nombre:<10} alpha[{i:2d}]  PDF {pdf[i]:+.6g}"
                              f"{'' if esc == 1 else f' x{esc:.5g} = {pdf_esc[i]:+.6g}'}"
                              f"   codigo {cod[i]:+.6g}   <-- DIFIERE")
    if total == 0:
        print("\nNO SE COMPARO NADA: los bloques de la tabla no se pudieron leer")
        return 1
    print(f"\n{total} valores comparados contra la Tabla III: {fallos} difieren, "
          f"{ilegibles} ilegibles sin confirmar, {confirmados} resueltos "
          f"mirando la pagina")
    return 1 if (fallos or ilegibles) else 0


if __name__ == "__main__":
    sys.exit(main())
