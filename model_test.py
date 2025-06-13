import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import ResNet18, Residual
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

def val_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = r'dataset\val'

    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])
    # 定义数据集处理方法变量
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    val_data = ImageFolder(ROOT_TRAIN, transform=val_transform)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return val_dataloader


from tqdm import tqdm
import torch

def val_model_process(model, val_dataloader, classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    num_classes = len(classes)
    # 初始化混淆矩阵：num_classes x num_classes 的0矩阵
    cm = [[0]*num_classes for _ in range(num_classes)]

    val_corrects = 0
    val_num = 0

    with torch.no_grad():
        for val_data_x, val_data_y in tqdm(val_dataloader, desc="valing", unit="batch"):
            val_data_x = val_data_x.to(device)
            val_data_y = val_data_y.to(device)
            output = model(val_data_x)
            pre_lab = torch.argmax(output, dim=1)

            for t, p in zip(val_data_y.cpu().numpy(), pre_lab.cpu().numpy()):
                cm[t][p] += 1
                if t == p:
                    val_corrects += 1
            val_num += val_data_x.size(0)

    val_acc = val_corrects / val_num
    print(f"测试准确率为：{val_acc:.4f}")

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
    val_dataloader = val_data_process()

    # 计算整体测试准确率
    classes = ['cat', 'dog']
    val_dataloader = val_data_process()
    val_model_process(model, val_dataloader, classes)

    # 下面是单张图片预测代码，可以保留或删掉
    from PIL import Image
    image = Image.open('cc.jpg')

    normalize = transforms.Normalize([0.48607032,0.45353173,0.4160252], [0.06886391,0.06542894,0.0667423])
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        classes = ['cat', 'dog']  # 根据你的类别修改
        print("预测值：",  classes[pre_lab.item()])



