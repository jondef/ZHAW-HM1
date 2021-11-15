
"""
@author: IT19ta_WIN / zahlesev@students.zhaw.ch
@version: 1.0, 25.01.2021
"""


import numpy as np

"""==================== INPUT ===================="""
A = np.array([[1, 1, 0], [3, -1, 2], [2, -1, 3]], dtype=np.float64)

v0 = np.array([1, 0, 0]).T
"""==============================================="""

def von_mises_iteration(A_in, v_in, iterations):
    A = np.copy(A_in)
    v = np.copy(v_in)
    eigv = 0

    for i in range(iterations):
        v_next = (A @ v) / (np.linalg.norm(A @ v))
        eigv = (v.T @ A @ v) / (v.T @ v)
        v = v_next

    return eigv, v


print(np.linalg.eig(A))
ew, ev = von_mises_iteration(A, v0, 1000)
print("Grösster Eigenwert aka Spektralradius = " + str(ew))
print("Zugehöriger Eigenvektor = " + str(ev))

