import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import ResNet18, Residual
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = r'dataset\test'

    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader


from tqdm import tqdm
import torch

def test_model_process(model, test_dataloader, classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_classes = len(classes)
    # 初始化混淆矩阵：num_classes x num_classes 的0矩阵
    cm = [[0]*num_classes for _ in range(num_classes)]

    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in tqdm(test_dataloader, desc="Testing", unit="batch"):
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)

            for t, p in zip(test_data_y.cpu().numpy(), pre_lab.cpu().numpy()):
                cm[t][p] += 1
                if t == p:
                    test_corrects += 1
            test_num += test_data_x.size(0)

    test_acc = test_corrects / test_num
    print(f"测试准确率为：{test_acc:.4f}")

    # 打印混淆矩阵
    print("混淆矩阵（行：真实标签，列：预测标签）：")
    print("\t" + "\t".join(classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]}\t" + "\t".join(str(x) for x in row))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # 加载模型
    model = ResNet18(Residual)
    model.load_state_dict(torch.load('best_gln_model.pth', map_location=device))
    model = model.to(device)

    # 加载测试数据
    test_dataloader = test_data_process()

    # 计算整体测试准确率
    classes = ['cat', 'dog']
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader, classes)

    # 下面是单张图片预测代码，可以保留或删掉
    from PIL import Image
    image = Image.open('cc.jpg')

    normalize = transforms.Normalize([0.48607032,0.45353173,0.4160252], [0.06886391,0.06542894,0.0667423])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        classes = ['cat', 'dog']  # 根据你的类别修改
        print("预测值：",  classes[pre_lab.item()])



