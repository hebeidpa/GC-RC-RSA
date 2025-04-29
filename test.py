import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from PIL import Image

# 参数设定
model_path = ''
data_dir = ''  # ⚠️ 更换为新验证集路径
output_csv = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset 类（与训练时一致）
class PatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = sorted(os.listdir(data_dir))
        self.file_list = []
        self.label_list = []
        for label in self.labels:
            label_path = os.path.join(data_dir, label)
            for file in os.listdir(label_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.file_list.append(os.path.join(label_path, file))
                    self.label_list.append(self.labels.index(label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.label_list[index], self.file_list[index]

# 模型结构（必须与训练一致）
class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=8, num_features=2048):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = F.relu(self.resnet(x))
        logits = self.classifier(features)
        return features, logits

# 指标计算
def calculate_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    sensitivities, specificities = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        sensitivities.append(sens)
        specificities.append(spec)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return auc, acc, np.mean(sensitivities), np.mean(specificities)

# Bootstrap置信区间
def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstrap=1000):
    aucs, accs, sens_list, spec_list = [], [], [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        auc, acc, sens, spec = calculate_metrics(
            np.array(y_true)[idx], np.array(y_pred)[idx], np.array(y_prob)[idx]
        )
        aucs.append(auc)
        accs.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)
    def ci(values): return (np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5))
    return {
        'AUC': ci(aucs),
        'ACC': ci(accs),
        'SENS': ci(sens_list),
        'SPEC': ci(spec_list)
    }

# 数据加载
dataset = PatchDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

# 加载模型
model = FeatureExtractor().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 验证
all_preds, all_probs, all_labels, all_paths = [], [], [], []

with torch.no_grad():
    for images, labels, paths in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        _, outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)

# 原始指标
auc, acc, sens, spec = calculate_metrics(all_labels, all_preds, all_probs)
print(f'Validation Results:')
print(f'  AUC : {auc:.4f}')
print(f'  ACC : {acc:.4f}')
print(f'  SENS: {sens:.4f}')
print(f'  SPEC: {spec:.4f}')

# Bootstrap估计
print('\nBootstrap Validation (n=1000):')
boot = bootstrap_metrics(all_labels, all_preds, all_probs, n_bootstrap=1000)
for key, (mean, low, high) in boot.items():
    print(f'  {key:<5}: {mean:.4f} (95% CI: {low:.4f} - {high:.4f})')

# 保存CSV
df = pd.DataFrame({
    'filepath': all_paths,
    'true_label': all_labels,
    'predicted_label': all_preds,
    'confidence': np.max(all_probs, axis=1)
})
df.to_csv(output_csv, index=False)
print(f'预测结果保存至：{output_csv}')
