import pandas as pd
import numpy as np

def entropy(col):
    unique_vals, counts = np.unique(col, return_counts=True)
    total = len(col)
    prob = counts / total
    return -np.sum(prob * np.log2(prob))

def joint_entropy(col1, col2):
    combined = list(zip(col1, col2))
    unique_combos, combo_counts = np.unique(combined, return_counts=True, axis=0)
    total = len(combined)
    prob = combo_counts / total
    return -np.sum(prob * np.log2(prob))


df1 = pd.read_excel(r'C:\Users\16775\Desktop\信息论\data1.xls')
df2 = pd.read_excel(r'C:\Users\16775\Desktop\信息论\data2.xls')

# 处理第一个表格
X1 = df1.iloc[:, 0]
Y1 = df1.iloc[:, 1]

# 计算信息熵 H(X) for df1
H_X1 = entropy(X1)
H_X1_float = float(H_X1)
print(f"H(X) for data1.xls = {H_X1_float:.6f}")  # 显示6位小数的浮点数

# 计算信息熵 H(Y) for df1
H_Y1 = entropy(Y1)
H_Y1_float = float(H_Y1)
print(f"H(Y) for data1.xls = {H_Y1_float:.6f}")

# 计算联合熵 H(X,Y) for df1
H_XY1 = joint_entropy(X1, Y1)
H_XY1_float = float(H_XY1)
print(f"H(X,Y) for data1.xls = {H_XY1_float:.6f}")

# 计算条件熵 H(X|Y) for df1
H_X_given_Y1 = H_XY1 - H_Y1
H_X_given_Y1_float = float(H_X_given_Y1)
print(f"H(X|Y) for data1.xls = {H_X_given_Y1_float:.6f}")

# 计算条件熵 H(Y|X) for df1
H_Y_given_X1 = H_XY1 - H_X1
H_Y_given_X1_float = float(H_Y_given_X1)
print(f"H(Y|X) for data1.xls = {H_Y_given_X1_float:.6f}")

# 计算互信息 I(X,Y) for df1
I_XY1 = H_X1 + H_Y1 - H_XY1
I_XY1_float = float(I_XY1)
print(f"I(X,Y) for data1.xls = {I_XY1_float:.6f}\n")

# 处理第二个表格
X2 = df2.iloc[:, 0]
Y2 = df2.iloc[:, 1]

# 计算信息熵 H(X) for df2
H_X2 = entropy(X2)
H_X2_float = float(H_X2)
print(f"H(X) for data2.xls = {H_X2_float:.6f}")

# 计算信息熵 H(Y) for df2
H_Y2 = entropy(Y2)
H_Y2_float = float(H_Y2)
print(f"H(Y) for data2.xls = {H_Y2_float:.6f}")

# 计算联合熵 H(X,Y) for df2
H_XY2 = joint_entropy(X2, Y2)
H_XY2_float = float(H_XY2)
print(f"H(X,Y) for data2.xls = {H_XY2_float:.6f}")

# 计算条件熵 H(X|Y) for df2
H_X_given_Y2 = H_XY2 - H_Y2
H_X_given_Y2_float = float(H_X_given_Y2)
print(f"H(X|Y) for data2.xls = {H_X_given_Y2_float:.6f}")

# 计算条件熵 H(Y|X) for df2
H_Y_given_X2 = H_XY2 - H_X2
H_Y_given_X2_float = float(H_Y_given_X2)
print(f"H(Y|X) for data2.xls = {H_Y_given_X2_float:.6f}")

# 计算互信息 I(X,Y) for df2
I_XY2 = H_X2 + H_Y2 - H_XY2
I_XY2_float = float(I_XY2)
print(f"I(X,Y) for data2.xls = {I_XY2_float:.6f}")