import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from MyModule.SEBlock import SEBlock
from model import ResNet18, Residual



# ========== STEP 2: Build model and load weights ==========
def build_model(weight_path=None,se_block=False):
    """
    Build model architecture and return different models based on the type
    """
    if not weight_path:
        print("Error: NO weight path")
        return None
    if se_block:
        model = ResNet18(Residual,attention=SEBlock)
    else:
        model = ResNet18(Residual)

    # Load weights
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model


# ========== STEP 3: Test data preprocessing ==========
def test_data_process():
    ROOT_TEST = r'dataset/test'
    normalize = transforms.Normalize([0.48607032, 0.45353173, 0.4160252], [0.06886391, 0.06542894, 0.0667423])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    test_data = ImageFolder(ROOT_TEST, transform=test_transform)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0)
    return test_dataloader, test_data.classes


# ========== STEP 4: Model evaluation ==========
def test_model_process(model, test_dataloader, classes, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_data_x, test_data_y in tqdm(test_dataloader, desc="testing", unit="batch"):
            test_data_x = test_data_x.to(device)
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)

            all_preds.extend(pre_lab.cpu().numpy())
            all_labels.extend(test_data_y.numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (rows: true labels, columns: predicted labels):")
    print(cm)

    # Visualize and save confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix image saved to: {save_path}")
    else:
        plt.show()

    # Overall accuracy
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test accuracy: {test_acc:.4f}")


# ========== STEP 5: Main function ==========
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model_pwd = "model"
    pth_list = os.listdir(Path(model_pwd))
    for pth_name in pth_list:
        model_weight_path = str(Path(model_pwd) / pth_name)
        if model_weight_path.find("False") != -1:
            model = build_model(model_weight_path,False)
        else:
            model = build_model(model_weight_path,True)
        print(model_weight_path)
        model = model.to(device)

        test_dataloader, classes = test_data_process()
        save_path = f"ConfusionMatrix/confusion_matrix_{pth_name}.png"
        test_model_process(model, test_dataloader, classes, save_path=save_path)
