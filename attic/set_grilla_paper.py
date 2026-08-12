s = open("main.py").read()
a = '"airspeed_norm": np.linspace(0.4, 2.0, 81, dtype=np.float32),'
b = '"airspeed_norm": np.linspace(0.9, 2.0, 41, dtype=np.float32),'
c = "            np.deg2rad(-10), np.deg2rad(40), 80, dtype=np.float32"
d = "            np.deg2rad(-40), np.deg2rad(20), 60, dtype=np.float32"
assert a in s and c in s, "no encontre las lineas a cambiar"
open("main.py", "w").write(s.replace(a, b).replace(c, d))
print("grilla cambiada a la del paper 1")
