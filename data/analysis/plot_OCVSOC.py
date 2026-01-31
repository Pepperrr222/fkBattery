import pandas as pd
import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'battery_usage', 'master.csv')
# 读取CSV文件
df = pd.read_csv(file_path)

# 按phone_test_id分组，提取每组中battery_cell为Cell01的行
Cell01 = df[df['battery_cell'] == 'Cell01']
Cell01 = Cell01[Cell01['battery_state_label'] == 'new']
Cell01 = Cell01[Cell01['battery_dataset'] == 3]

# 按phone_test_id排序
Cell01 = Cell01.sort_values('phone_test_id')

print(Cell01)

# 可选：保存结果到新文件
Cell01.to_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'Cell01_data.csv'), index=False)