s = open("escenario.py").read()
s = s.replace('DE_PULL = np.deg2rad(float(A[7]) if len(A) > 7 else -13.0)',
'''DE_PULL = np.deg2rad(float(A[7]) if len(A) > 7 else -15.0)
# forma del elevador optimo, medida sobre la propia politica:
#   +15 deg durante 0.19 s | pico -14.9 deg, dura 0.30 s | sostiene -6.2 deg
D_PUSH, D_PULL = 0.19, 0.30
DE_HOLD = np.deg2rad(-6.2)''')
s = s.replace('''            de = float(DE_PUSH) if t_uns is None else float(DE_PULL)''',
'''            dtr = t - t_det                      # tres escalones que siguen
            if dtr < D_PUSH:                     # la forma del optimo
                de = float(DE_PUSH)
            elif dtr < D_PUSH + D_PULL:
                de = float(DE_PULL)
            else:
                de = float(DE_HOLD)''')
s = s.replace('"piloto: +15 deg hasta el "\n      "nose-down, despues %.0f deg fijo\\n"\n      % (np.rad2deg(DE_NOREAC), np.rad2deg(DE_PULL))',
              '"piloto escalonado: +15 (%.2f s), %.0f (%.2f s), %.1f\\n"\n'
              '      % (np.rad2deg(DE_NOREAC), D_PUSH, np.rad2deg(DE_PULL), D_PULL, np.rad2deg(DE_HOLD))')
s = s.replace('"3_maniobras/escenario_v%03d.png"', '"3_maniobras/escenario_pd_v%03d.png"')
s = s.replace('"\\n$V_0=%.2f\\\\,V_s$, elevador del piloto escalonado" % (T_DP, T_PIL, V0)',
              '"\\n$V_0=%.2f\\\\,V_s$, piloto escalonado siguiendo la forma del DP" % (T_DP, T_PIL, V0)')
open("escenario.py", "w").write(s)
print("piloto reconstruido con la forma del DP")
