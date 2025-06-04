import os
from shutil import copy
import random
from copy import copy
import random

import cv2
import numpy as np
from tqdm import tqdm  # 进度条


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def create_data():
    # 获取data文件夹下所有文件夹名（即需要分类的类名）
    file_path = 'catanddog'
    flower_class = [cla for cla in os.listdir(file_path)]

    # 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
    mkfile('dataset/train')
    for cla in flower_class:
        mkfile('dataset/train/' + cla)

    # 创建 验证集val 文件夹，并由类名在其目录下创建子目录
    mkfile('dataset/test')
    for cla in flower_class:
        mkfile('dataset/test/' + cla)

    # 划分比例，训练集 : 测试集 = 9 : 1
    split_rate = 0.1

    # 遍历所有类别的全部图像并按比例分成训练集和验证集
    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
        images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
        for index, image in enumerate(images):
            # eval_index 中保存验证集val的图像名称
            if image in eval_index:
                image_path = cla_path + image
                new_path = 'dataset/test/' + cla
                copy(image_path, new_path)  # 将选中的图像复制到新路径

            # 其余的图像保存在训练集train中
            else:
                image_path = cla_path + image
                new_path = 'dataset/train/' + cla
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()
    print("processing done!")
def change_image():
    cat_folder = './catanddog/cat/'
    dog_folder = './catanddog/dog/'
    cat_list = os.listdir(cat_folder)
    dog_list = os.listdir(dog_folder)
    for i in range(6000):
        if os.path.exists(cat_folder+cat_list[i]):
            os.remove(cat_folder+cat_list[i])
        if os.path.exists(dog_folder+dog_list[i]):
            os.remove(dog_folder+dog_list[i])
    print("chang_image_done")
def calculate_image_mean(src):
    # 设置数据集主文件夹路径
    dataset_path = src
    # 初始化累加变量
    sum_pixels = np.zeros(3, dtype=np.float64)  # B, G, R 通道的和
    sum_squared_pixels = np.zeros(3, dtype=np.float64)  # B, G, R 通道的平方和
    num_pixels = 0  # 记录总像素数
    # 遍历所有子文件夹
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):  # 过滤掉非文件夹
            continue
        # 遍历该类别文件夹下的所有图片
        image_files = [f for f in os.listdir(subdir_path) if f.endswith((".jpg", ".png"))]
        for image_file in tqdm(image_files, desc=f"Processing {subdir}"):
            image_path = os.path.join(subdir_path, image_file)
            image = cv2.imread(image_path)  # 读取 BGR 图片
            if image is None:
                print(f"⚠️ 无法读取图片: {image_path}")
                continue  # 跳过无效图片
            image = image / 255.0  # 归一化到 [0,1]
            # 累加 BGR 通道的像素值
            sum_pixels += np.sum(image, axis=(0, 1))
            sum_squared_pixels += np.sum(image ** 2, axis=(0, 1))
            num_pixels += image.shape[0] * image.shape[1]  # 计算总像素数

    # 确保数据有效，防止除以 0
    if num_pixels == 0:
        raise ValueError("错误: 没有可用的图片数据，无法计算均值和方差。")

    # 计算均值
    mean = sum_pixels / num_pixels

    # 计算方差（variance = E[x^2] - (E[x])^2）
    variance = (sum_squared_pixels / num_pixels) - (mean ** 2)

    # OpenCV 读取的图像是 BGR 格式，转换为 RGB 格式
    mean = mean[::-1]
    variance = variance[::-1]

    print(f"数据集均值 (RGB): {mean}")
    print(f"数据集方差 (RGB): {variance}")
if __name__ == "__main__":
    # change_image()
    # create_data()
    calculate_image_mean('dataset/train')