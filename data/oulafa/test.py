import pandas as pd

file_name = 'master.csv'

try:
    # 只读取 0 行数据，专门看表头
    df = pd.read_csv(file_name, nrows=0)
    cols = df.columns.tolist()

    print("\n" + "=" * 50)
    print(f"读取成功！文件 '{file_name}' 的所有列名如下：")
    print("-" * 50)

    for i, col in enumerate(cols):
        # 打印序号和列名，同时检查是否有隐藏的空格
        print(f"[{i:02d}]  '{col}'")

    print("-" * 50)
    print("你可以直接把上面的输出结果发给我。")
    print("=" * 50 + "\n")

except Exception as e:
    print(f"❌ 读取失败: {e}")