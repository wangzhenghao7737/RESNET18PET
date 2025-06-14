import os
from shutil import copy
from copy import copy
import random
import cv2
import numpy as np
from tqdm import tqdm


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def create_data():
    # Get all folder names under 'catanddog' (i.e., class names)
    file_path = 'catanddog'
    flower_class = [cla for cla in os.listdir(file_path)]

    # Create training set folder and class subfolders
    mkfile('dataset/train')
    for cla in flower_class:
        mkfile('dataset/train/' + cla)

    # Create test set folder and class subfolders
    mkfile('dataset/test')
    for cla in flower_class:
        mkfile('dataset/test/' + cla)

    # Split ratio: training : test = 9 : 1
    split_rate = 0.1

    # Traverse all images and split into training and test sets
    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'  # subdirectory for a class
        images = os.listdir(cla_path)  # image file names
        num = len(images)
        etest_index = random.sample(images, k=int(num * split_rate))   # randomly sample k images for test set
        for index, image in enumerate(images):
            # images in etest_index go to test set
            if image in etest_index:
                image_path = cla_path + image
                new_path = 'dataset/test/' + cla
                copy(image_path, new_path)  # copy image to new path

            # others go to training set
            else:
                image_path = cla_path + image
                new_path = 'dataset/train/' + cla
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # progress bar
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
    sum_pixels = np.zeros(3, dtype=np.float64)  # B, G, R channel sums
    sum_squared_pixels = np.zeros(3, dtype=np.float64) # B, G, R squared sums
    num_pixels = 0  # total number of pixels
    # Traverse all subfolders
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):    # skip non-directories
            continue
        # Traverse all images in this class folder
        image_files = [f for f in os.listdir(subdir_path) if f.endswith((".jpg", ".png"))]
        for image_file in tqdm(image_files, desc=f"Processing {subdir}"):
            image_path = os.path.join(subdir_path, image_file)
            image = cv2.imread(image_path)   # read image in BGR
            if image is None:
                print(f"⚠ Unable to read image: {image_path}")
                continue  # skip invalid image
            image = image / 255.0   # normalize to [0,1]
            # accumulate BGR values
            sum_pixels += np.sum(image, axis=(0, 1))
            sum_squared_pixels += np.sum(image ** 2, axis=(0, 1))
            num_pixels += image.shape[0] * image.shape[1]   # count total pixels

    # Ensure data is valid to avoid division by zero
    if num_pixels == 0:
        raise ValueError("Error: No valid image data, cannot compute mean and variance.")

    # Calculate mean
    mean = sum_pixels / num_pixels

    # Calculate variance (variance = E[x^2] - (E[x])^2)
    variance = (sum_squared_pixels / num_pixels) - (mean ** 2)

    # Convert BGR to RGB
    mean = mean[::-1]
    variance = variance[::-1]

    print(f"Dataset mean (RGB): {mean}")
    print(f"Dataset variance (RGB): {variance}")
if __name__ == "__main__":
    # change_image()
    # create_data()
    calculate_image_mean('dataset/train')