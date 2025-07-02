"""
coding:utf-8
* @Author:jhxiao
* @name:convert_label_rgb.py
* @Time:2024/9/21 11:05
* @Description:Description of this file
"""
import os
import numpy as np
from PIL import Image

# 标签到 RGB 颜色的映射字典
label_to_color = {
    0: (0, 0, 0),         # 黑色
    1: (255, 255, 0),     # 黄色 (255, 255, 255) CD的需要转化
    2: (255, 0, 0),       # 红色
    3: (128, 0, 128),     # 紫色
    4: (0, 255, 0),       # 绿色
    5: (0, 0, 255)        # 蓝色
}

# 输入文件夹路径和输出文件夹路径
input_folder = 'E:\\data\\mydata3\\B_label'  # 输入文件夹路径
output_folder = 'E:\\data\\mydata3\\B_label_RGB'  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图像文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.png'):  # 根据实际情况修改图像格式
        # 读取标签图像
        label_image_path = os.path.join(input_folder, file_name)
        label_image = Image.open(label_image_path)
        label_array = np.array(label_image)

        # 创建 RGB 图像的空白数组
        rgb_image = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

        # 将标签值映射为对应的 RGB 颜色
        for label, color in label_to_color.items():
            rgb_image[label_array == label] = color

        # 保存 RGB 图像
        rgb_image = Image.fromarray(rgb_image)
        rgb_image.save(os.path.join(output_folder, file_name))

print("转换完成！")