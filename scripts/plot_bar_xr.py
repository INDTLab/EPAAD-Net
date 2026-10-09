
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
excel_file = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'docs', 'XR.xlsx')
df = pd.read_excel(excel_file)

# 删除第一列
df = df.drop(df.columns[0], axis=1)

# 将数据分为两组，每组三列
group1 = df.iloc[:, :3]
group2 = df.iloc[:, 3:]

# 绘制第一组分组柱状图
fig1, ax1 = plt.subplots(figsize=(10, 6))  # 设置画布大小

num_cols = len(group1.columns)
num_groups = len(group1)
bar_width = 0.2  # 柱子宽度
group_width = bar_width * 3  # 每组的宽度
group_offsets = np.linspace(-group_width/2, group_width/2, num_groups)  # 每组的偏移量

# 每一列为一个分组
for i, col in enumerate(group1.columns):
    x = np.arange(num_groups) + group_offsets[i]
    y = group1[col]
    ax1.bar(x, y, width=bar_width, label=col)

# 设置x轴标签为行名
ax1.set_xticks(range(num_groups))
ax1.set_xticklabels(group1.index)

# 添加图例
ax1.legend()

# 添加标题和标签
plt.title('Grouped Bar Chart - Group 1')
plt.xlabel('Groups')
plt.ylabel('Values')

# 调整布局以防止x轴标签被裁切
plt.tight_layout()

# 显示图表
plt.show()

# 绘制第二组分组柱状图
fig2, ax2 = plt.subplots(figsize=(10, 6))  # 设置画布大小

num_cols = len(group2.columns)
num_groups = len(group2)
bar_width = 0.2  # 柱子宽度
group_width = bar_width * num_cols  # 每组的宽度
group_offsets = np.linspace(-group_width/2, group_width/2, num_groups)  # 每组的偏移量

# 每一列为一个分组
for i, col in enumerate(group2.columns):
    x = np.arange(num_groups) + group_offsets[i]
    y = group2[col]
    ax2.bar(x, y, width=bar_width, label=col)

# 设置x轴标签为行名
ax2.set_xticks(range(num_groups))
ax2.set_xticklabels(group2.index)

# 添加图例
ax2.legend()

# 添加标题和标签
plt.title('Grouped Bar Chart - Group 2')
plt.xlabel('Groups')
plt.ylabel('Values')

# 调整布局以防止x轴标签被裁切
plt.tight_layout()

# 显示图表
plt.show()
