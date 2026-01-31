import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'battery_usage', 'CS2_33_12_16_10_2.csv')
# Read the CSV file
df = pd.read_csv(file_path)

# Extract relevant columns
cycle_index = df['Cycle_Index'].values
charge_capacity = df['Charge_Capacity(Ah)'].values
discharge_capacity = df['Discharge_Capacity(Ah)'].values

# Calculate C_effective: difference between consecutive charge capacities
C_effective = charge_capacity[1:] - charge_capacity[:-1]

# cycle_index is already a NumPy array (from .values). Slicing returns an ndarray, so do NOT call .values on it.
# Convert to float to be safe and ensure lengths match.
k = np.asarray(cycle_index[:-1], dtype=float)
if k.shape[0] != C_effective.shape[0]:
    raise ValueError(f"Length mismatch: k has {k.shape[0]} elements but C_effective has {C_effective.shape[0]} elements")

# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(k, C_effective, 'b-', label='C_effective', linewidth=1.5)

# Linear fitting: C_n = C_0(1 - k*N)
# Rearrange: C_n = C_0 - C_0*N*k (linear form: y = a + b*x)
def linear_model(k, C0, N):
    return C0 * (1 - k * N)

# Fit the model
popt, _ = curve_fit(linear_model, k, C_effective)
C0, N = popt

# Plot the fitted curve
fitted_curve = linear_model(k, C0, N)
plt.plot(k, fitted_curve, 'r--', label=f'Fit: C_n = {C0:.4f}(1 - {N:.6f}*k)', linewidth=2)

plt.xlabel('Cycle Index (N)')
plt.ylabel('Effective Charge Capacity (Ah)')
plt.title('Battery Capacity Degradation')
plt.legend()
plt.grid(True, alpha=0.3)
output_path = os.path.join(ROOT_DIR, 'problem1', 'figs', 'capacity_degradation.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Fitted parameters:")
print(f"C_0 = {C0:.6f} Ah")
print(f"k = {N:.10f}")