
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import os

# 数据
N = 6  # 每组柱子的数量
F1 = (0.9944, 0.9412, 0.9548, 0.92, 0.9495, 0.8162)  # 男性组的平均值
F1_1d = (0.8606, 0.8, 0.9314, 0.9056, 0.9012, 0.8091)  # 女性组的平均值
F1_PA = (0.841, 0.8205, 0.9327, 0.8743, 0.8367, 0.8087)
F1_AE = (0.9944, 0.8421, 0.9165, 0.905, 0.7711, 0.8155)
F1_sin = (0.9655, 0.8421, 0.9105, 0.8841, 0.7789, 0.8143)
F1_cos = (0.9892, 0.8205, 0.9291, 0.9133, 0.7664, 0.8131)
F1_AE_att = (0.9522, 0.8205, 0.9289, 0.8985, 0.8832, 0.8138)

ind = np.arange(N)  # 每组柱子的索引
width = 0.1  # 柱子宽度

# 创建图表
fig, ax = plt.subplots(figsize=(15, 6))

# 绘制柱状图
rects1 = ax.bar(ind - 2.5*width, F1, width, label='F1', color='#5F66CB', zorder=2)
rects2 = ax.bar(ind - 1.5*width, F1_1d, width, label='F1_1d', color='#ADD8E6', zorder=2)
rects3 = ax.bar(ind - 0.5*width, F1_PA, width, label='F1_SCTM', color='#00D6B1', zorder=2)
rects4 = ax.bar(ind + 0.5*width, F1_AE, width, label='F1_AE', color='#808A87', zorder=2)
rects5 = ax.bar(ind + 1.5*width, F1_sin, width, label='F1_sin', color='#E3CF57', zorder=2)
rects6 = ax.bar(ind + 2.5*width, F1_cos, width, label='F1_cos', color='#FFDAB9', zorder=2)
rects7 = ax.bar(ind + 3.5*width, F1_AE_att, width, label='F1_AE_att', color='#FFB6C1', zorder=2)

# 设置Y轴范围
ax.set_ylim(0.7, 1)

# 添加标签、标题等
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('SMD', 'NAB', 'MBA', 'SMAP', 'MSL', 'SWaT'))
ax.legend()

# # 添加数值标签
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

# 添加网格
ax.grid(True, zorder=0)

outfolder = r'C:\Users/14141\Desktop\各种版本的图'

image_file = os.path.join(outfolder, f'消融实验-改名后.png')
plt.savefig(image_file, bbox_inches='tight', dpi=300)

plt.show()
plt.close()  # 关闭图以释放资源