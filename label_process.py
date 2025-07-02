"""
coding:utf-8
* @Author:jhxiao
* @name:label_process.py
* @Time:2024/9/12 10:22
* @Description:Description of this file 
"""
import os
import cv2
import numpy as np
import os
from PIL import Image
import numpy as np

# 定义文件夹路
phase1_folder = r"E:\data\data\A_label_RGB"  # 时相1图像文件夹路径
phase2_folder = r"E:\data\data\B_label_RGB"  # 时相2图像文件夹路径
change_label_folder = r"E:\data\data\CD_label_RGB_1"  # 变化标签图保存路径
processed_phase1_folder = r"E:\data\data\A_label_SCD_RGB"  # 处理后时相1保存路径
processed_phase2_folder = r"E:\data\data\B_label_SCD_RGB"  # 处理后时相2保存路径

# 确保保存的文件夹存在
os.makedirs(change_label_folder, exist_ok=True)
os.makedirs(processed_phase1_folder, exist_ok=True)
os.makedirs(processed_phase2_folder, exist_ok=True)

# 遍历时相1文件夹中的图片
for filename in os.listdir(phase1_folder):
    if filename.lower().endswith(".png"):  # 确保文件是png格式，不区分大小写
        phase1_path = os.path.join(phase1_folder, filename)
        phase2_path = os.path.join(phase2_folder, filename)

        # 检查时相2文件夹中是否存在同名文件
        if not os.path.exists(phase2_path):
            print(f"时相2文件夹中找不到对应的文件：{filename}")
            continue

        # 使用PIL读取两张时相图
        phase1_img = Image.open(phase1_path).convert('RGB')  # 转换为RGB格式
        phase2_img = Image.open(phase2_path).convert('RGB')  # 转换为RGB格式

        # 转换为numpy数组
        phase1_array = np.array(phase1_img)
        phase2_array = np.array(phase2_img)

        # 检查图像尺寸是否一致
        if phase1_array.shape != phase2_array.shape:
            print(f"图像尺寸不一致：{filename}")
            continue

        # 初始化变化标签图，黑色和白色，分别对应RGB (0, 0, 0) 和 (255, 255, 255)
        change_label_img = np.zeros_like(phase1_array)  # 黑色为 (0, 0, 0)
        change_label_img.fill(255)  # 填充为白色

        # 处理两张时相图
        processed_phase1 = phase1_array.copy()
        processed_phase2 = phase2_array.copy()

        # 逐像素比较
        for i in range(phase1_array.shape[0]):
            for j in range(phase1_array.shape[1]):
                if np.array_equal(phase1_array[i, j], phase2_array[i, j]):
                    # 像素相同，变化标签图生成黑色
                    change_label_img[i, j] = [0, 0, 0]
                    # # 两张时相图对应像素生成黑色
                    processed_phase1[i, j] = [0, 0, 0]
                    processed_phase2[i, j] = [0, 0, 0]
                else:
                    # 像素不同，保持原有像素
                    change_label_img[i, j] = [255, 255, 255]

        # 转换回Image格式并保存
        change_label_img_pil = Image.fromarray(change_label_img)
        processed_phase1_pil = Image.fromarray(processed_phase1)
        processed_phase2_pil = Image.fromarray(processed_phase2)

        # 保存变化标签图
        change_label_img_pil.save(os.path.join(change_label_folder, filename))
        # 保存处理后的时相1和时相2图
        processed_phase1_pil.save(os.path.join(processed_phase1_folder, filename))
        processed_phase2_pil.save(os.path.join(processed_phase2_folder, filename))

print("处理完成！")
