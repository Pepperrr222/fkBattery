import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 设置路径（对应你的真实文件）
# ==========================================
FILE_PATH = r'C:\Users\24119\Desktop\MCM\fkBattery\data\oulafa\CS2_33_12_16_10.csv'


def run_euler_simulation():
    # 读取 CSV
    print("正在读取原始实验数据...")
    df = pd.read_csv(FILE_PATH)

    # 清理列名空格
    df.columns = df.columns.str.strip()

    # ==========================================
    # 2. 设定电池物理参数 (根据 CALCE 电池标准设定)
    # ==========================================
    Q_nominal = 1.1  # 额定容量 (Ah)
    R0 = 0.05  # 假设内阻 (Ohm)
    initial_soc = 1.0  # 假设从满电开始

    # 假设一组通用的锂电池 OCV-SOC 曲线系数 (V = c0 + c1*s + c2*s^2...)
    # 如果你有具体拟合好的系数，可以替换这里
    coeffs = [3.5, 0.5, -0.2, 0.1, 0.05, -0.01]

    # ==========================================
    # 3. 提取 CSV 中的数据列
    # ==========================================
    # 对应你 CSV 里的真实列名
    currents = df['Current(A)'].values
    voltages = df['Voltage(V)'].values
    times = df['Test_Time(s)'].values

    # 结果存储
    sim_socs = [initial_soc]
    predicted_voltages = []

    print(f"开始欧拉法迭代计算，总计 {len(df)} 个数据点...")

    # ==========================================
    # 4. 欧拉迭代 (核心逻辑)
    # ==========================================
    soc = initial_soc

    for i in range(len(df)):
        # --- A. 计算当前预测电压 (用于验证模型) ---
        # V_pred = OCV(soc) - I * R0
        vocv = sum(c * (soc ** j) for j, c in enumerate(coeffs))
        v_pred = vocv - currents[i] * R0
        predicted_voltages.append(v_pred)

        # --- B. 欧拉法更新 SOC ---
        # 对应微分方程: dSOC/dt = -I / Q
        # 离散化: SOC(n+1) = SOC(n) - (I * dt) / (Q * 3600)
        if i < len(df) - 1:
            dt = times[i + 1] - times[i]
            # 这里的 dt 是两行数据之间的时间差

            # 注意：如果电流为正表示放电，则减；如果为负表示充电，则加
            soc = soc - (currents[i] * dt) / (Q_nominal * 3600)

            # 限制 SOC 范围在 0-1 之间
            soc = max(0.0, min(1.0, soc))
            sim_socs.append(soc)

    # ==========================================
    # 5. 可视化产出
    # ==========================================
    print("计算完成，生成结果图...")

    plt.figure(figsize=(12, 5))

    # 左图：SOC 随时间的变化
    plt.subplot(1, 2, 1)
    plt.plot(times, sim_socs, 'g-', label='Estimated SOC (Euler)')
    plt.title('SOC Recovery using Euler Method')
    plt.xlabel('Test Time (s)')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)

    # 右图：模拟电压 vs 真实电压 (验证模型准确度)
    plt.subplot(1, 2, 2)
    plt.plot(times, voltages, 'k', alpha=0.4, label='Actual Voltage (CSV)')
    plt.plot(times, predicted_voltages, 'r--', label='Predicted Voltage')
    plt.title('Voltage Prediction Accuracy')
    plt.xlabel('Test Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_euler_simulation()