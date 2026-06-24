import pandas as pd

# 读取 CSV 文件
file_path = r"D:\PythonProject\MVCM_project\data_all.csv"
df = pd.read_csv(file_path)

# 统计 type 列中各类别数量
type_counts = df["type"].value_counts().sort_index()

# 输出结果
print("各类别样本数量：")
print(type_counts)