import pandas as pd

# 你的文件路径
file_path = r'C:\Users\24119\Desktop\MCM\fkBattery\data\oulafa\master.csv'

try:
    # 1. 尝试读取前两行数据
    df_sample = pd.read_csv(file_path, sep=None, engine='python', nrows=2)

    print("--- 诊断结果 ---")
    # 2. 打印原始列名列表（带引号，为了看清有没有空格）
    print("1. 原始列名清单:")
    print([col for col in df_sample.columns.tolist()])

    # 3. 打印第一行数据的部分内容
    print("\n2. 第一行数据的具体内容 (检查值是否匹配):")
    print(df_sample.iloc[0].to_dict())

    # 4. 检查分隔符
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        print("\n3. CSV 文件的第一行原始文本:")
        print(first_line.strip())

except Exception as e:
    print(f"解析失败: {e}")