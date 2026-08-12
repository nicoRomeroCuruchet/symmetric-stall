s = open("deteccion.py").read()
# el mando durante el retardo pasa a ser parametro
s = s.replace('TAU_M = float(sys.argv[5]) if len(sys.argv) > 5 else 0.50',
              'TAU_M = float(sys.argv[5]) if len(sys.argv) > 5 else 0.50\n'
              'DE_NOREAC = np.deg2rad(float(sys.argv[6]) if len(sys.argv) > 6 else -25.0)')
s = s.replace('de = float(get_optimal_action(obs, pi)[0][0]) if detectado else 0.0',
              'de = float(get_optimal_action(obs, pi)[0][0]) if detectado else float(DE_NOREAC)')
s = s.replace('print("antes de detectar: de=0, palanca=0 (nadie hace nada)\\n")',
              'print("antes de detectar: de=%.1f deg (el piloto TIRA, no reacciono), palanca=0\\n"\n'
              '      % np.rad2deg(DE_NOREAC))')
s = s.replace('("3_maniobras/deteccion_v%03d.png" % round(V0 * 100))',
              '("3_maniobras/deteccion_v%03d_de%+03d.png" % (round(V0*100), round(np.rad2deg(DE_NOREAC))))')
s = s.replace('" — $V_0=%.2f\\\\,V_s$" % (TAU_DP, TAU_H, TAU_M, V0)',
              '"\\n$V_0=%.2f\\\\,V_s$, sin reaccionar $\\\\delta_e=%.0f^\\\\circ$" % (TAU_DP, TAU_H, TAU_M, V0, np.rad2deg(DE_NOREAC))')
open("deteccion.py", "w").write(s)
print("parcheado")
