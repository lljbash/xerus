import numpy as np
import matplotlib.pyplot as plt

errs = np.load("errors.LegendrePolynomials.npy")
# errs = np.load("errors.HermitePolynomials.npy")
plt.plot(errs[0], 'ro-', label="CM")
plt.plot(errs[1], 'ko-', label="Ref")
plt.legend()
plt.tight_layout()
plt.show()
