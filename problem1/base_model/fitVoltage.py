import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------
# 1. 读取数据
# ---------------------------------------------------------
# 请将 'your_data.csv' 替换为你实际的文件名
file_path = os.path.join(ROOT_DIR, 'data', 'processed', 'OCV_SOC.csv')

try:
    # 尝试读取 CSV，如果出现编码错误，可以尝试 encoding='gbk'
    df = pd.read_csv(file_path)
    
    # 检查列名，防止全角/半角符号问题
    # 你的列名示例：SOC(％), OCV_charge(V), OCV_discharge(V), OCV_ave(V)
    # 这里做一个简单的列名映射，确保代码能找到数据
    print("CSV 列名:", df.columns.tolist())
    
    # 找到 SOC 和 OCV_ave 对应的列（模糊匹配，兼容不同符号）
    soc_col = [c for c in df.columns if 'SOC' in c][0]
    ocv_col = [c for c in df.columns if 'OCV_ave' in c][0]
    
    # 原始 SOC 列（可能以百分比保存）和 OCV 列
    x_raw = pd.to_numeric(df[soc_col], errors='coerce').values
    y_all = pd.to_numeric(df[ocv_col], errors='coerce').values

except Exception as e:
    print(f"读取文件出错: {e}")
    # 为了演示，生成一段模拟数据
    print("生成模拟数据进行演示...")
    x_raw = np.linspace(0.1, 1.0, 100)
    y_all = 3.2 + 0.01 * x_raw + 0.1 * np.log(x_raw)

# ---------------------------------------------------------
# 2. 数据预处理
# ---------------------------------------------------------

# 将 SOC 标准化为分数（0.0 - 1.0）。如果原始值最大大于 1.5，则视为百分比（0-100）
# 否则视为分数（0.0 - 1.0）
if np.nanmax(x_raw) > 1.5:
    x_frac_all = x_raw / 100.0
else:
    x_frac_all = x_raw.copy()

# 只保留 SOC(Ah) 对应 0.0 - 0.8 的点
mask_range = (x_frac_all >= 0.0) & (x_frac_all <= 0.8)
# 去除 NaN 并应用范围过滤
mask = (~np.isnan(x_frac_all)) & (~np.isnan(y_all)) & mask_range
x_data = x_frac_all[mask]
y_data = y_all[mask]

# 保留全部用于绘图的原始（分数形式）数据，供展示和调试
x_all_for_plot = x_frac_all
y_all_for_plot = y_all

# 检查用于拟合的数据点数量
if len(x_data) < 5:
    raise ValueError(f"可用于拟合的数据点太少: {len(x_data)}。请检查 SOC 范围或数据完整性。")

print(f"Using {len(x_data)} points for fit, SOC range: {x_data.min():.4f} - {x_data.max():.4f} (fraction)")

# ---------------------------------------------------------
# 3. 定义电压模型
# ---------------------------------------------------------
def voltage_model(x, a_log, a_lin, a_exp, k, C):
    """
    V = a_log * ln(x) + a_lin * x + a_exp * exp(k * x) + C
    x: SOC (0.0 - 1.0)
    """
    # 加上一个极小值 epsilon 防止 log(0)
    epsilon = 1e-6
    
    term_log = a_log * np.log(x + epsilon)
    term_linear = a_lin * x
    term_exp = a_exp * np.exp(k * (x - 1)) # 技巧：用 (x-1) 让指数项在 SOC=1 时为 1，数值更稳定
    # 如果你坚持用 exp(kx)，只需把上面的 (x-1) 改为 x，但 k 值可能会很大或很小
    
    return term_log + term_linear + term_exp + C

# ---------------------------------------------------------
# 4. 执行拟合
# ---------------------------------------------------------
# 初始猜测 (Initial Guess) - 对于包含指数的方程非常重要
# a_log: 通常为正 (0.1 ~ 1.0)
# a_lin: 线性斜率 (0.5 ~ 2.0)
# a_exp: 指数项系数 (0.01 ~ 0.5)
# k: 指数增长率 (1 ~ 5)
# C: 截距 (3.0 ~ 4.0)
p0 = [0.1, 0.5, 0.05, 2.0, 3.5]

try:
    popt, pcov = curve_fit(voltage_model, x_data, y_data, p0=p0, maxfev=10000)
    
    # 提取最佳参数
    a_log_opt, a_lin_opt, a_exp_opt, k_opt, C_opt = popt
    
    print("-" * 30)
    print("拟合结果:")
    print(f"a_log    = {a_log_opt:.6f}")
    print(f"a_linear = {a_lin_opt:.6f}")
    print(f"a_exp    = {a_exp_opt:.6f}")
    print(f"k        = {k_opt:.6f}")
    print(f"C        = {C_opt:.6f}")
    print("-" * 30)
    
    # 计算拟合优度 R^2
    residuals = y_data - voltage_model(x_data, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2 (拟合优度): {r_squared:.6f}")

    # ---------------------------------------------------------
    # 5. 绘图结果
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 原始全部数据（以灰色展示）和用于拟合的数据（高亮）
    plt.scatter(x_all_for_plot * 100, y_all_for_plot, s=10, color='lightgrey', alpha=0.6, label='Original Data')
    plt.scatter(x_data * 100, y_data, s=20, color='blue', alpha=0.8, label='Used for Fit')

    # 拟合曲线（在 0-1 的区间绘制）
    x_fit = np.linspace(0, 1, 200)
    y_fit = voltage_model(x_fit, *popt)
    plt.plot(x_fit * 100, y_fit, color='red', linewidth=2, label='Fitted Curve')
    
    plt.title(f'OCV-SOC Curve Fitting (SOC in 0-0.8 used) (R^2={r_squared:.4f})')
    plt.xlabel('State of Charge (SOC) %')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

except Exception as e:
    print(f"拟合失败: {e}")
    print("建议：检查数据中是否有异常值，或者调整 p0 初始猜测值。")