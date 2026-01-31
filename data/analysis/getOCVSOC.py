import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

# 读取CSV文件（使用相对于项目根的路径）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'battery_usage', 'CS2_33_12_16_10.csv')
df = pd.read_csv(file_path)

# 提取所需列
test_time = df['Test_Time(s)']
charge_capacity = df['Charge_Capacity(Ah)']
discharge_capacity = df['Discharge_Capacity(Ah)']
voltage = df['Voltage(V)']

# 计算SOC和OCV
Q_total = 0.907158595176865
SOC = (charge_capacity - discharge_capacity) / Q_total
OCV = voltage

# 分割充电和放电数据 (1-199行为充电，200-404行为放电)
# 注意：行号从0开始，所以1-199对应索引0-198，200-404对应索引199-403
charge_data = df.iloc[0:199]
discharge_data = df.iloc[199:404]

# 计算各段的SOC和OCV
SOC_charge = (charge_data['Charge_Capacity(Ah)'] - charge_data['Discharge_Capacity(Ah)']) / Q_total
OCV_charge = charge_data['Voltage(V)']

SOC_discharge = (discharge_data['Charge_Capacity(Ah)'] - discharge_data['Discharge_Capacity(Ah)']) / Q_total
OCV_discharge = discharge_data['Voltage(V)']

# 仅保留 SOC 在 [0,1] 的点
mask_charge = (SOC_charge >= 0.0) & (SOC_charge <= 1.0)
mask_discharge = (SOC_discharge >= 0.0) & (SOC_discharge <= 1.0)
SOC_charge = SOC_charge[mask_charge].reset_index(drop=True)
OCV_charge = OCV_charge[mask_charge].reset_index(drop=True)
SOC_discharge = SOC_discharge[mask_discharge].reset_index(drop=True)
OCV_discharge = OCV_discharge[mask_discharge].reset_index(drop=True)

# 去除 NaN 并按 SOC 排序，为插值做准备
df_charge = pd.DataFrame({'SOC': SOC_charge, 'OCV': OCV_charge}).dropna().sort_values('SOC')
df_discharge = pd.DataFrame({'SOC': SOC_discharge, 'OCV': OCV_discharge}).dropna().sort_values('SOC')
SOC_charge = df_charge['SOC'].values
OCV_charge = df_charge['OCV'].values
SOC_discharge = df_discharge['SOC'].values
OCV_discharge = df_discharge['OCV'].values

# 创建图表
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 充电曲线
axes[0].plot(SOC_charge, OCV_charge, 'b-', linewidth=2, label='Charge')
axes[0].set_xlabel('SOC', fontsize=12)
axes[0].set_ylabel('OCV (V)', fontsize=12)
axes[0].set_title('Charge OCV-SOC Curve', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 放电曲线
axes[1].plot(SOC_discharge, OCV_discharge, 'r-', linewidth=2, label='Discharge')
axes[1].set_xlabel('SOC', fontsize=12)
axes[1].set_ylabel('OCV (V)', fontsize=12)
axes[1].set_title('Discharge OCV-SOC Curve', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 平均OCV曲线
# 计算公共的SOC范围（同时限制在 [0,1] 内）
if len(SOC_charge) == 0 or len(SOC_discharge) == 0:
    raise ValueError("充电或放电段在 SOC∈[0,1] 内没有数据点，无法计算平均 OCV。")
soc_min = max(SOC_charge.min(), SOC_discharge.min(), 0.0)
soc_max = min(SOC_charge.max(), SOC_discharge.max(), 1.0)
if soc_max <= soc_min:
    raise ValueError(f"没有重叠的 SOC 范围用于平均 (min={soc_min}, max={soc_max})")
SOC_common = np.linspace(soc_min, soc_max, 100)

# 在相同的SOC点上插值并求平均（按相同 SOC 对应 OCV 求平均）
OCV_charge_interp = np.interp(SOC_common, SOC_charge, OCV_charge)
OCV_discharge_interp = np.interp(SOC_common, SOC_discharge, OCV_discharge)
OCV_avg = (OCV_charge_interp + OCV_discharge_interp) / 2

axes[2].plot(SOC_common, OCV_avg, 'g-', linewidth=2, label='Average')
axes[2].plot(SOC_charge, OCV_charge, 'b--', alpha=0.5, label='Charge')
axes[2].plot(SOC_discharge, OCV_discharge, 'r--', alpha=0.5, label='Discharge')
axes[2].set_xlabel('SOC', fontsize=12)
axes[2].set_ylabel('OCV (V)', fontsize=12)
axes[2].set_title('Average OCV-SOC Curve', fontsize=14)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
# 确保输出目录存在并保存图像到 data/analysis/figs
figs_dir = os.path.join(ROOT_DIR, 'data', 'analysis', 'figs')
proc_dir = os.path.join(ROOT_DIR, 'data', 'processed')
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(proc_dir, exist_ok=True)
fig_path = os.path.join(figs_dir, 'OCV_SOC_curve.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')

# 保存平均数据到 CSV（SOC 使用百分比表示）
out_df = pd.DataFrame({
    'SOC(％)': (SOC_common * 100),
    'OCV_charge(V)': OCV_charge_interp,
    'OCV_discharge(V)': OCV_discharge_interp,
    'OCV_ave(V)': OCV_avg
})
csv_path = os.path.join(proc_dir, 'OCV_SOC.csv')
out_df.to_csv(csv_path, index=False)

plt.show()
print(f"Saved figure to: {fig_path}")
print(f"Saved CSV to: {csv_path}")

# 打印统计信息
print("充电数据统计:")
print(f"  SOC范围: {SOC_charge.min():.4f} - {SOC_charge.max():.4f}")
print(f"  OCV范围: {OCV_charge.min():.4f} - {OCV_charge.max():.4f} V")
print("\n放电数据统计:")
print(f"  SOC范围: {SOC_discharge.min():.4f} - {SOC_discharge.max():.4f}")
print(f"  OCV范围: {OCV_discharge.min():.4f} - {OCV_discharge.max():.4f} V")