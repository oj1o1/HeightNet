import scipy.io as scio
import numpy as np
import math

dataFile = r"D:\MatlabProject\Single-View-Metrology-master\All_Parameters_test8.mat"
data = scio.loadmat(dataFile)

Vanishing_X, Vanishing_Y, t0, b0, Reference_Height, Vz = map(np.squeeze, [data['vX'], data['vY'], data['Z_O'], data['Origin'], data['len_Z_O'], data['vZ']])

horizon = np.cross(Vanishing_Y, Vanishing_X) / np.linalg.norm(horizon)

b, r = np.array([656, 42, 1]), np.array([656, 135, 1])

line1 = np.cross(b0, b)
v = np.cross(line1, horizon) / v[2]

line2 = np.cross(v, t0)
vertical_line = np.cross(r, b)
t = np.cross(line2, vertical_line) / t[2]

height = Reference_Height * math.sqrt(np.sum((r - b)**2)) * math.sqrt(np.sum((Vz - t)**2)) / (math.sqrt(np.sum((t - b)**2)) * math.sqrt(np.sum((Vz - r)**2))

print("height =", height)
