"""Compare the code's tables against Table III of Riley's PDF.

Until now the checks verified that the CUDA kernel agreed with the Python
model. That says nothing about whether either of them says what the report
says: the two copies can agree and both be wrong, which is exactly what was
happening.

This reads papers/riley.pdf, extracts Table III and compares value by value
against the code's arrays. The text comes from a 1985 scan and the OCR confuses
characters systematically -- commas for periods, O for zero, Z for two, q for
four -- so it is normalised before comparing, and anything unreadable is
reported as ILLEGIBLE rather than guessed.

Usage:  PYTHONPATH=. python verificar_tablas_riley.py
"""
import re
import subprocess
import sys

import numpy as np

from symmetric_stall.aircraft.symmetric_full_grumman import SymmetricFullGrumman

PDF = "papers/Riley.pdf"
DEG_TO_RAD = 57.2958  # the code stores control derivatives per radian

# Table III block -> (first line, last line) in the pdftotext -layout dump,
# and which column each group of three is.
BLOCKS = [
    # (start line, end line, [(group_name, attr_ct0, attr_ct05)])
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

# Control derivatives are tabulated PER DEGREE and the code stores them per
# radian; the q and alpha-dot ones multiply an already dimensionless quantity
# and go as they are. CD_de2 additionally carries the 1e-2 from its column
# header.
ESCALA = {
    "_CL_DE_TABLE_CT0": DEG_TO_RAD, "_CL_DE_TABLE_CT05": DEG_TO_RAD,
    "_CM_DE_TABLE_CT0": DEG_TO_RAD, "_CM_DE_TABLE_CT05": DEG_TO_RAD,
    "_CD_DE_TABLE_CT0": DEG_TO_RAD, "_CD_DE_TABLE_CT05": DEG_TO_RAD,
    "_CD_DE2_TABLE_CT0": 1e-2 * DEG_TO_RAD ** 2,
    "_CD_DE2_TABLE_CT05": 1e-2 * DEG_TO_RAD ** 2,
}


# ---------------------------------------------------------------------------
# Tablas LATERALES, III(d) fuerza lateral, III(e) guiniada, III(f) alabeo.
# They only exist on the 6- and 8-DOF branches; on the 4-DOF they are skipped.
#
# The shape repeats in all three: two blocks of three groups. In the first
# block the third group (the aileron derivative) has NO C_T split and occupies
# a single column, so that row carries 8 tokens and not 9.
BLOCKS_LAT = [
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

# The CONTROL and BETA derivatives are tabulated per degree; the p and r ones
# multiply an already dimensionless quantity and go as they are.
for _n in ("_CY_BETA_TABLE_CT0", "_CY_BETA_TABLE_CT05", "_CY_DA_TABLE",
           "_CY_DR_TABLE_CT0", "_CY_DR_TABLE_CT05",
           "_CN_BETA_TABLE_CT0", "_CN_BETA_TABLE_CT05", "_CN_DA_TABLE",
           "_CN_DR_TABLE_CT0", "_CN_DR_TABLE_CT05",
           "_CL_ROLL_BETA_TABLE_CT0", "_CL_ROLL_BETA_TABLE_CT05",
           "_CL_ROLL_DA_TABLE", "_CL_ROLL_DR_TABLE"):
    ESCALA[_n] = DEG_TO_RAD


# Entries where the OCR is unrecoverable and which were resolved by looking at
# the PDF PAGE as an image. The value is what is read there; the script reports
# them as confirmed rather than as differences, with the page alongside so that
# cualquiera pueda repetir la lectura.
#
# The pattern is always the same: the scan's "q" is a 9 (-,349q = -.3499,
# -*Olq3 = -.0193), and leading zeros swallow the decimal point.
CONFIRMADOS = {
    ("_CL_DE_TABLE_CT05", 1): (0.0134, "pag. 32, CL_de CT=0.5, alpha=-5"),
    ("_CD_O_TABLE_CT05", 1): (-0.3499, "pag. 33, CD_o CT=0.5, alpha=-5"),
    ("_CD_DE2_TABLE_CT05", 0): (0.0000, "pag. 33, CD_(de)2 CT=0.5, alpha=-10"),
    ("_CM_O_TABLE", 0): (0.2700, "pag. 34, Cm_o CT=0.0, alpha=-10"),
    ("_CM_DE_TABLE_CT0", 2): (-0.0193, "pag. 34, Cm_de CT=0.0, alpha=0"),
    ("_CM_DE_TABLE_CT05", 13): (-0.0153, "pag. 34, Cm_de CT=0.5, alpha=40"),
    ("_CM_Q_TABLE_CT05", 10): (-19.0600, "pag. 34, Cm_q CT=0.5, alpha=25"),
    ("_CM_ADOT_TABLE_CT05", 9): (0.0000, "pag. 34, Cm_adot CT=0.5, alpha=20"),
    # --- lateral ones, resolved against pages 35, 36 and 37 ---
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

# systematic OCR confusions over digits
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
            f"COULD NOT READ {PDF}. Without the report there is nothing to "
            f"compare against, and passing green would be worse than failing.")
    return r.stdout.split("\n")


def numbers(line):
    """Return a row's numeric tokens, or None where it cannot be read.

    The leading decimal point is what degrades most: ".0051" comes out as
    "o0051", "*0350" or ",0062". It is normalised separately, because mapping
    it as
    digito convierte 0.0051 en 51 y eso parece un error de transcripcion
    when it is a reading error.
    """
    # the CD(de)2 column repeats the header's "x10-2" on the alpha=-10 row,
    # glued to the values, which are additionally left without a separator
    line = re.sub(r"x[lI1][oO0]-?2?", " ", line)
    out = []
    for tok in line.split():
        # the scan leaves stray page marks ("I", "0") in the margin; taking
        # them for numbers shifts ALL the columns of that row by one. But an
        # illegible value can also end up with no digits at all ("-.nl_"), and
        # that one DOES occupy a column: they are told apart by length.
        if not any(ch.isdigit() for ch in tok):
            if len(tok) >= 3 and any(c in "._" for c in tok):
                out.append(None)
            continue
        sign = ""
        body = tok
        if body[:1] in "-+":
            sign, body = body[0], body[1:]
        if body[:1] in "oO0Cc*,.°•" and not body.startswith("0."):
            body = "." + body[1:]
        t = (sign + body).translate(OCR)
        # un token legible es sign opcional, digitos y a lo sumo un punto
        if re.fullmatch(r"[-+]?\d*\.?\d+", t) and any(ch.isdigit() for ch in t):
            out.append(float(t))
        elif re.search(r"[\d._]", tok):
            out.append(None)            # something numeric but illegible
    return out


def main():
    lineas = leer_pdf()
    a = SymmetricFullGrumman()
    total = failures = ilegibles = confirmados = 0
    print("Table III of the PDF against the code arrays\n")

    for start, end, ncol, groups in [(a_, b_, 9, c_) for a_, b_, c_ in BLOCKS] + BLOCKS_LAT:
        rows = []
        for l in lineas[start - 1:end]:
            if not l.strip():
                continue
            v = numbers(l)
            if len(v) >= ncol:
                rows.append(v[:ncol])
        if len(rows) != 14:
            print(f"  !! block {start}-{end}: read {len(rows)} rows, expected 14")
            continue

        M = np.array([[np.nan if x is None else x for x in f] for f in rows])
        # with 8 columns the last group carries only one, so the index of the
        # third group is not 3*gi but whatever is left
        for gi, (name, at0, at05) in enumerate(groups):
            base = 3 * gi + 1
            for col, attr in ((base, at0), (base + 1, at05)):
                if attr is None or not hasattr(a, attr) or col >= M.shape[1]:
                    continue
                pdf = M[:, col]
                cod = np.asarray(getattr(a, attr), dtype=np.float64)
                esc = ESCALA.get(attr, 1.0)
                pdf_esc = pdf * esc
                bad = np.isnan(pdf)
                diff = np.abs(pdf_esc - cod) > np.maximum(1e-4 * np.abs(cod), 1e-7)
                diff &= ~bad
                total += 14
                for i in np.where(bad | diff)[0]:
                    ref = CONFIRMADOS.get((attr, i))
                    if ref is not None and abs(ref[0] * esc - cod[i]) <= max(
                            1e-6 * abs(cod[i]), 1e-9):
                        confirmados += 1
                        print(f"  {name:<10} alpha[{i:2d}]  OCR illegible, "
                              f"confirmed against the image: {ref[0]:+.6g}"
                              f"  ({ref[1]})")
                        continue
                    if bad[i]:
                        ilegibles += 1
                        print(f"  {name:<10} alpha[{i:2d}]  ILLEGIBLE and NO "
                              f"confirmar, codigo {cod[i]:+.6g}")
                    else:
                        failures += 1
                        print(f"  {name:<10} alpha[{i:2d}]  PDF {pdf[i]:+.6g}"
                              f"{'' if esc == 1 else f' x{esc:.5g} = {pdf_esc[i]:+.6g}'}"
                              f"   codigo {cod[i]:+.6g}   <-- DIFIERE")
    if total == 0:
        print("\nNOTHING WAS COMPARED: the table blocks could not be read")
        return 1
    print(f"\n{total} values compared against Table III: {failures} differ, "
          f"{ilegibles} unconfirmed illegibles, {confirmados} resueltos "
          f"by looking at the page")
    return 1 if (failures or ilegibles) else 0


if __name__ == "__main__":
    sys.exit(main())
