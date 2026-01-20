# 手写中文 OCR（PyTorch + OpenVINO）

Handwritten Chinese Text Recognition using CNN + CTC, supporting training and inference with PyTorch and OpenVINO.

面向手写中文文本行识别的参考实现，支持 CNN + CTC 训练与推理。

## 目录 / Table of Contents
- [环境要求](#环境要求)
- [数据准备与预处理](#数据准备与预处理)
- [本地训练完整流程](#本地训练完整流程)
- [测试与评估](#测试与评估)
- [Colab 快速开始](#colab-快速开始)
- [部署推理](#部署推理openvinonox)
- [许可证与引用](#许可证)

---

## 环境要求

**训练环境：**
- Python 3.8+
- PyTorch 1.8.1+ (GPU 版本推荐，CUDA 11+)
- NVIDIA Apex（可选，用于混合精度 AMP）
- 其他依赖：`pip install -r requirements.txt`

**推理/部署环境（可选）：**
- Intel OpenVINO 2021.3+

```bash
# 克隆项目
git clone https://github.com/AndrewCullacino/handwritten-chinese-ocr-samples.git
cd handwritten-chinese-ocr-samples

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

---

## 数据准备与预处理

### 1. 获取 CASIA-HWDB 数据集

从 [CASIA-HWDB 官网](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) 下载：
- **HWDB2.0** / **HWDB2.1** / **HWDB2.2**（文本行数据，含 Train 和 Test）

下载后目录结构：
```
/path/to/hwdb-raw/
├── HWDB2.0Train/
│   └── *.dgrl
├── HWDB2.0Test/
│   └── *.dgrl
├── HWDB2.1Train/
│   └── *.dgrl
├── HWDB2.1Test/
│   └── *.dgrl
├── HWDB2.2Train/
│   └── *.dgrl
└── HWDB2.2Test/
    └── *.dgrl
```

### 2. 预处理：从 DGRL 提取 PNG 图像

使用 `dgrl2png.py` 将 DGRL 文件转为 PNG 图像和标注文件：

```bash
# 设置路径变量
RAW_DATA=/path/to/hwdb-raw
OUTPUT_DIR=data/hwdb2.x

# 创建输出目录
mkdir -p $OUTPUT_DIR/train $OUTPUT_DIR/val $OUTPUT_DIR/test

# 提取 HWDB2.0 训练数据
python utils/casia-hwdb-data-preparation/dgrl2png.py \
    $RAW_DATA/HWDB2.0Train \
    $OUTPUT_DIR/train \
    --image_height 128

# 提取 HWDB2.1 训练数据（追加到同一目录）
python utils/casia-hwdb-data-preparation/dgrl2png.py \
    $RAW_DATA/HWDB2.1Train \
    $OUTPUT_DIR/train \
    --image_height 128

# 提取 HWDB2.2 训练数据（追加到同一目录）
python utils/casia-hwdb-data-preparation/dgrl2png.py \
    $RAW_DATA/HWDB2.2Train \
    $OUTPUT_DIR/train \
    --image_height 128

# 提取所有测试数据
python utils/casia-hwdb-data-preparation/dgrl2png.py \
    $RAW_DATA/HWDB2.0Test \
    $OUTPUT_DIR/test \
    --image_height 128

python utils/casia-hwdb-data-preparation/dgrl2png.py \
    $RAW_DATA/HWDB2.1Test \
    $OUTPUT_DIR/test \
    --image_height 128

python utils/casia-hwdb-data-preparation/dgrl2png.py \
    $RAW_DATA/HWDB2.2Test \
    $OUTPUT_DIR/test \
    --image_height 128
```

**dgrl2png.py 参数说明：**
| 参数 | 说明 |
|------|------|
| `source` | DGRL 文件/目录/zip 文件路径 |
| `target` | 输出目录 |
| `--image_height` | 输出图像高度（默认 128） |

每个目录会生成 `dgrl_img_gt.txt`，格式为 `filename.png,标注文本`。

### 3. 划分 Train/Val 并生成元数据

预处理完成后，需要：
1. 合并所有 `dgrl_img_gt.txt`
2. 划分 train/val（建议 90%/10%）
3. 生成字符表 `chars_list.txt`

```python
# prepare_dataset.py
import os
import random
import shutil

DATA_DIR = 'data/hwdb2.x'

# 1. 读取所有训练样本
train_gt_file = f'{DATA_DIR}/train/dgrl_img_gt.txt'
with open(train_gt_file, 'r', encoding='utf-8') as f:
    all_train_lines = [line.strip() for line in f if line.strip()]

# 2. 读取测试样本
test_gt_file = f'{DATA_DIR}/test/dgrl_img_gt.txt'
with open(test_gt_file, 'r', encoding='utf-8') as f:
    test_lines = [line.strip() for line in f if line.strip()]

# 3. 随机划分 train/val (90%/10%)
random.seed(42)
random.shuffle(all_train_lines)
val_size = int(len(all_train_lines) * 0.1)
val_lines = all_train_lines[:val_size]
train_lines = all_train_lines[val_size:]

# 4. 移动 val 图像到 val 目录
os.makedirs(f'{DATA_DIR}/val', exist_ok=True)
for line in val_lines:
    img_name = line.split(',')[0]
    src = f'{DATA_DIR}/train/{img_name}'
    dst = f'{DATA_DIR}/val/{img_name}'
    if os.path.exists(src):
        shutil.move(src, dst)

# 5. 写入元数据文件
with open(f'{DATA_DIR}/train_img_id_gt.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_lines))

with open(f'{DATA_DIR}/val_img_id_gt.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_lines))

with open(f'{DATA_DIR}/test_img_id_gt.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_lines))

# 6. 生成字符表
all_chars = set()
for line in train_lines + val_lines + test_lines:
    if ',' in line:
        text = line.split(',', 1)[1]
        all_chars.update(text)

with open(f'{DATA_DIR}/chars_list.txt', 'w', encoding='utf-8') as f:
    for char in sorted(all_chars):
        f.write(char + '\n')

print(f'Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}')
print(f'Characters: {len(all_chars)}')
```

运行脚本：
```bash
python prepare_dataset.py
```

### 4. 最终数据目录结构

```
data/hwdb2.x/
├── train/
│   └── *.png                   # 训练图像
├── val/
│   └── *.png                   # 验证图像
├── test/
│   └── *.png                   # 测试图像
├── train_img_id_gt.txt         # 训练标注（img_name,text）
├── val_img_id_gt.txt           # 验证标注
├── test_img_id_gt.txt          # 测试标注
└── chars_list.txt              # 字符表（每行一个字符）
```

---

## 本地训练完整流程

### 1. 开始训练

```bash
python main.py -m hctr \
    -d data/hwdb2.x \
    -b 8 \
    -ep 50 \
    -lr 1e-4 \
    -pf 100 \
    -vf 5000 \
    -j 4 \
    --gpu 0
```

**参数说明：**
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `-m` | 模型类型 | `hctr` |
| `-d` | 数据集路径 | `data/hwdb2.x` |
| `-b` | batch size | A100: 16-24, V100: 8-12, T4: 4-8 |
| `-ep` | 训练轮数 | 30-50 |
| `-lr` | 学习率 | 1e-4 |
| `-pf` | 打印频率（每 N batch） | 100 |
| `-vf` | 验证频率（每 N batch） | 5000 |
| `-j` | 数据加载线程数 | 4 |
| `--gpu` | 指定 GPU | 0（单卡）或不指定（多卡） |

### 2. 监控训练

训练过程会自动保存：
- `hctr_checkpoint.pth.tar` - 最新 checkpoint
- `hctr_best_acc_XX.XX.pth.tar` - 最佳验证精度模型

### 3. 恢复训练

如果训练中断，可从 checkpoint 恢复：
```bash
python main.py -m hctr \
    -d data/hwdb2.x \
    -b 8 \
    -ep 100 \
    -re hctr_checkpoint.pth.tar \
    --gpu 0
```

### 4. GPU 显存优化建议

| GPU | 显存 | 推荐 batch size |
|-----|------|----------------|
| A100 | 40GB | 16-24 |
| V100 | 16GB | 8-12 |
| T4 | 16GB | 4-8 |
| RTX 3090 | 24GB | 12-16 |

如遇 OOM，可降低 batch size 或启用梯度累积。

---

## 测试与评估

### 单张图片推理

```bash
python test.py -m hctr \
    -f hctr_best_acc_XX.XX.pth.tar \
    -i test_image.png \
    -dm greedy-search
```

### 测试集批量评估

```bash
python test.py -m hctr \
    -f hctr_best_acc_XX.XX.pth.tar \
    -i data/hwdb2.x \
    -bm \
    -dm greedy-search \
    -b 16 \
    -pf 20
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `-f` | 模型文件路径 |
| `-i` | 输入图像或数据集目录 |
| `-bm` | 基准测试模式（批量评估） |
| `-dm` | 解码方式：`greedy-search` 或 `beam-search` |
| `-b` | batch size |
| `-pf` | 打印频率 |

### Beam Search + 语言模型（提升精度）

```bash
# 使用 Transformer 语言模型
python test.py -m hctr -f <model> -i <input> -bm \
    -dm beam-search \
    --use-tfm-pred \
    --transformer-path <tfm_model>

# 使用 n-gram 语言模型（低延迟）
python test.py -m hctr -f <model> -i <input> -bm \
    -dm beam-search \
    --skip-search \
    --kenlm-path <ngram.arpa>
```

---

## Colab 快速开始

如果你使用 Google Colab，可直接运行 `colab_train.ipynb`：

1. 将数据上传到 `My Drive/HWDB-data/`：
   ```
   HWDB2.0Train/, HWDB2.0Test/
   HWDB2.1Train/, HWDB2.1Test/
   HWDB2.2Train/, HWDB2.2Test/
   ```
2. 在 Colab 中选择 A100/V100/T4 GPU 运行时
3. 按顺序运行各单元格
4. 预处理结果会保存在 `My Drive/HWDB-data/preprocessed/`，下次可直接复用

---

## 部署推理（OpenVINO/ONNX）

### 1. 导出 ONNX
```bash
python utils/export_onnx.py
```

### 2. 转换为 OpenVINO IR（可选）
```bash
mo --input_model model.onnx --output_dir ./openvino_model
```

### 3. 使用 OpenVINO 推理
```bash
python deploy.py -lang hctr \
    -m openvino_model/model.xml \
    -dm greedy-search \
    -i test_image.png
```

### 4. 使用预训练模型（SCUT-EPT）

可使用 OpenVINO Model Zoo 的预训练模型：
```bash
# 下载预训练模型
python <openvino>/tools/downloader/downloader.py \
    --name handwritten-simplified-chinese-recognition-0001

# 推理
python deploy.py -lang hctr \
    -m <path-to-model.xml> \
    -dm greedy-search \
    -i <image>
```

---

## 项目结构

```
.
├── main.py                 # 训练入口
├── test.py                 # 测试/评估入口
├── deploy.py               # OpenVINO 推理
├── colab_train.ipynb       # Colab 训练笔记本
├── requirements.txt        # Python 依赖
├── models/
│   └── handwritten_ctr_model.py    # 模型定义
├── utils/
│   ├── dataset.py          # 数据加载
│   ├── ctc_codec.py        # CTC 编解码
│   ├── export_onnx.py      # ONNX 导出
│   └── casia-hwdb-data-preparation/
│       ├── dgrl2png.py     # DGRL → PNG 预处理
│       ├── gnt2png.py      # GNT → PNG 预处理
│       └── README.md
└── third-party/            # 语言模型训练
```

---

## 性能参考

在 ICDAR 2013 竞赛集上的字符错误率 (CER)：

| 方法 | 无语言模型 | 有语言模型 |
|------|-----------|-----------|
| LSTM-RNN-CTC | 16.50 | 11.60 |
| CNN-ResLSTM-CTC | 8.45 | 3.28 |
| **CNN-CTC-CBS (本项目)** | **6.38** | **2.49** |

---

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。

## 引用

```bibtex
@article{bliu2020hctr-cnn,
    Author = {Brian Liu, Xianchao Xu, Yu Zhang},
    Title = {Offline Handwritten Chinese Text Recognition with Convolutional Neural Networks},
    publisher = {arXiv},
    Year = {2020}
}
```

## 相关资源

- [论文：Offline Handwritten Chinese Text Recognition with CNNs (arXiv 2020)](https://arxiv.org/abs/2006.15619)
- [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/handwritten-simplified-chinese-recognition-0001)
- [CASIA-HWDB 数据集](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html)
