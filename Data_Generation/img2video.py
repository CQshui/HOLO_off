# -*- coding: utf-8 -*-
"""
@ Time:     2024/9/28 11:56 2024
@ Author:   CQshui
$ File:     img2video.py
$ Software: Pycharm
"""
from moviepy.editor import ImageSequenceClip
import os

def extract_number(s):
    return int(s.split('_')[1])  # 取第一个和第二个下划线之间的部分并转为整数


def jpg_to_mp4(input_folder, output_file, fps=30):
    """
    将指定文件夹中的所有JPG图片转换为MP4视频。

    参数:
    input_folder (str): 包含JPG图片的文件夹路径。
    output_file (str): 输出MP4视频的文件路径。
    fps (int): 每秒播放的帧数，默认是30。
    """
    # 获取文件夹中的所有JPG文件
    jpg_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # 确保图片文件存在
    if not jpg_files:
        raise ValueError(f"文件夹 {input_folder} 中没有找到任何JPG文件。")

    # 按文件名排序，确保图片按顺序排列
    jpg_files = sorted(jpg_files, key=extract_number)

    # 创建图片路径列表
    image_files = [os.path.join(input_folder, f) for f in jpg_files]

    print(f"找到 {len(image_files)} 张图片，开始转换为视频...")

    # 确保fps是有效的数字
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("帧率（fps）必须为正数。")

    # 创建视频剪辑
    clip = ImageSequenceClip(image_files, fps=fps)

    # 打印剪辑信息进行调试
    print(f"视频尺寸: {clip.size}, 帧率: {fps}")

    # 保存为MP4文件
    clip.write_videofile(output_file, codec='libx264', audio=False)


# 使用示例
input_folder = r'C:\Users\d1009\Desktop\test\temp\Mono8_0_Degree_16_2_0.bmp'  # 替换为你图片文件夹的路径
output_file = r'C:\Users\d1009\Desktop\test\temp\video\output.mp4'  # 输出视频的路径
jpg_to_mp4(input_folder, output_file, fps=2)


