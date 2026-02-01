import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 读取数据
file_path = 'master.csv'
df = pd.read_csv(file_path)

# 2. 预处理：按 phone_test_id 去重，只留 1000 条
df_mod = df.drop_duplicates(subset=['phone_test_id']).head(1000).copy()

# 定义 CPU 指数 gamma (CMOS 动态功耗通常与频率平方成正比)
gamma = 2.0

# --- 步骤 A: 独立分项拟合 (求 alpha) ---

# 1. 拟合屏幕：fit(Brightness, Display_ENERGY_UW) -> alpha_screen
# 注意：Display_ENERGY_UW 在你的表中是第 [70] 列
reg_screen = LinearRegression(fit_intercept=False)
X_screen = df_mod[['Brightness']].values
y_screen = df_mod['Display_ENERGY_UW'].values
reg_screen.fit(X_screen, y_screen)
alpha_screen = reg_screen.coef_[0]

# 2. 拟合CPU：fit(Freq^gamma, CPU_BIG_ENERGY_UW) -> alpha_cpu
# 使用大核频率 [52] 和大核功耗 [78]
reg_cpu = LinearRegression(fit_intercept=True)
X_cpu = (df_mod[['CPU_BIG_FREQ_KHz']] ** gamma).values
y_cpu = df_mod['CPU_BIG_ENERGY_UW'].values
reg_cpu.fit(X_cpu, y_cpu)
alpha_cpu = reg_cpu.coef_[0]

# 3. 拟合网络：fit(Bytes, WLANBT_ENERGY_UW) -> alpha_net
# 使用 WiFi 字节 [58] 和 WLAN 功耗 [73]
reg_net = LinearRegression(fit_intercept=True)
X_net = df_mod[['TOTAL_DATA_WIFI_BYTES']].values
y_net = df_mod['WLANBT_ENERGY_UW'].values
reg_net.fit(X_net, y_net)
alpha_net = reg_net.coef_[0]


# --- 步骤 B: 总功率线性回归 (求向量 w^T) ---
# 定义状态向量 x(t) = [u_screen, u_screen*Brightness, 1, freq^gamma, R, u_GPS, u_background]
# P_total_uW 在你的表中是第 [12] 列

X_total = pd.DataFrame()
# u_screen: 亮度大于0即为1
X_total['u_screen'] = (df_mod['Brightness'] > 0).astype(int)
# u_screen * Brightness
X_total['u_screen_brightness'] = X_total['u_screen'] * df_mod['Brightness']
# 常数项 1 (对应 beta_processor + beta_network + P_base)
X_total['ones'] = 1
# freq^gamma
X_total['freq_gamma'] = df_mod['CPU_BIG_FREQ_KHz'] ** gamma
# R (网络吞吐)
X_total['R'] = df_mod['TOTAL_DATA_WIFI_BYTES']
# u_GPS: GPS 功耗大于0即为1
X_total['u_GPS'] = (df_mod['GPS_ENERGY_UW'] > 0).astype(int)
# u_background: 用系统基础设施功耗是否大于0作为判定
X_total['u_background'] = (df_mod['INFRASTRUCTURE_ENERGY_UW'] > 0).astype(int)

y_total = df_mod['P_total_uW']

# 直接调用 sklearn 拟合向量 w^T
model_all = LinearRegression(fit_intercept=False)
model_all.fit(X_total, y_total)
w_T = model_all.coef_

# --- 4. 打印辨识结果 ---
print("="*60)
print("第一部分：分项拟合 Alpha 值 (单位: uW/unit)")
print("-"*60)
print(f"alpha_screen:    {alpha_screen:.6f}")
print(f"alpha_cpu:       {alpha_cpu:.6e}") # 频率数值大，系数可能很小
print(f"alpha_net:       {alpha_net:.6f}")

print("\n第二部分：总功率模型向量 w^T 辨识结果")
print("-"*60)
labels = [
    "w[0] (beta_screen)",
    "w[1] (alpha_screen*Area)",
    "w[2] (Combined Base Bias)",
    "w[3] (alpha_processor)",
    "w[4] (alpha_network)",
    "w[5] (alpha_GPS)",
    "w[6] (alpha_background)"
]
for label, val in zip(labels, w_T):
    print(f"{label.ljust(30)}: {val:.6f}")

print("="*60)
print(f"模型整体拟合优度 R^2: {model_all.score(X_total, y_total):.4f}")