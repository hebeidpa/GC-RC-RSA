
本项目主要包含图像分类模型的训练、测试以及特征提取三个主要功能，具体涵盖十折交叉验证训练、最终验证集评估、测试集评估和全切片图像（WSI）的特征提取与热力图生成。

## 代码说明

### 1. 训练脚本 (`train.py`)
#### 功能
使用十折交叉验证训练一个图像分类模型，并在最终验证集上进行评估。

#### 参数配置
- `output_dir`: 输出结果的保存路径。
- `num_classes`: 分类的类别数量。
- `batch_size`: 训练和验证时的批次大小。
- `num_epochs`: 训练的轮数。
- `lr`: 学习率。

#### 运行方式
```bash
python train.py
```

### 2. 测试脚本 (`test.py`)
#### 功能
加载训练好的模型，并在测试集上进行评估，保存测试结果。

#### 参数配置
- `output_dir`: 输出结果的保存路径。
- `model_path`: 训练好的模型的保存路径。
- `data_path`: 测试数据的路径。
- `num_classes`: 分类的类别数量。
- `batch_size`: 测试时的批次大小。

#### 运行方式
```bash
python test.py
```

### 3. 特征提取脚本 (`feature extractor.py`)
#### 功能
从全切片图像（WSI）中提取特征，并生成热力图。

#### 参数配置
- `WSI_FOLDER`: 全切片图像的文件夹路径。
- `weights_path`: 预训练模型的权重路径。
- `PATCH_SIZE`: 图像块的大小。
- `LEVEL`: 切片的级别。
- `STRIDE`: 滑动窗口的步长。
- `MODEL_TYPE`: 特征提取模型的类型。
- `feature_num`: 提取的特征数量。

#### 运行方式
```bash
python feature extractor.py
```

## 结果保存
- **训练结果**: 训练过程中的损失、准确率、AUC 等指标会保存为 `unipath_training_results_cv.csv`。
- **测试结果**: 测试集上的准确率、AUC、敏感度和特异度会保存为 `unipath_test_results.csv`。
- **特征提取结果**: 每个 WSI 的特征平均值会保存为 `wsi_feature_summary.csv`，同时每个 WSI 的热力图会保存为 `{slide_name}_heatmap.jpg`。

## 注意事项
- 请确保数据路径和模型权重路径正确，否则会导致程序运行出错。
- 在运行脚本之前，请检查是否已经安装了所有必要的依赖库。
```


