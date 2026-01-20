# 手写中文 OCR（PyTorch + OpenVINO）

面向手写中文文本行识别的参考实现，支持 CNN + CTC 训练与推理，包含 Colab 训练脚本与本地推理工具。

## 最新改动
- 支持同时使用 CASIA-HWDB 2.0 / 2.1 / 2.2 全量数据，预处理结果持久化在 `My Drive/HWDB-data/preprocessed`，重复运行会自动复用，避免重复耗时预处理。
- 训练数据与标签在宽度截断时保持同比例裁剪，保证 CTC 标签与图片同步（已提交到主代码）。
- Colab 笔记本 `colab_train.ipynb` 更新了数据校验、预处理、训练与恢复流程，可直接在 A100 环境高效训练。

## 环境要求
- Python 3.8+
- PyTorch 1.8.1+（建议使用 GPU，CUDA 11+）
- NVIDIA Apex（可选，用于 AMP）
- OpenVINO 2021.3+（仅部署/推理需要）

## 数据准备（Google Drive）
在 Google Drive 中放置原始 DGRL 数据：
```
My Drive/HWDB-data/
├── HWDB2.0Train/
├── HWDB2.0Test/
├── HWDB2.1Train/
├── HWDB2.1Test/
├── HWDB2.2Train/
└── HWDB2.2Test/
```

## Colab 快速开始（推荐）
1) 选择 A100/V100/T4 GPU 运行时。
2) 挂载 Drive，克隆项目并安装依赖。
3) 运行 `colab_train.ipynb`：
        - 数据校验：统计各版本 DGRL 文件数。
        - 预处理：若 `My Drive/HWDB-data/preprocessed` 已存在，会跳过并直接创建软链到 `data/hwdb2.x`；否则自动提取 2.0/2.1/2.2 全量数据并划分 train/val/test。
        - 训练：默认 `main.py -m hctr -d data/hwdb2.x`，按 GPU 规格自适应 batch size。
        - 评估：`test.py -m hctr -i data/hwdb2.x`。
        - 断线恢复：确保用第 11 步保存 checkpoint 到 Drive，之后第 12 步可用 `-re` 继续训练。

## 本地训练示例
```
python main.py -m hctr -d data/hwdb2.x -b 8 -pf 100 -lr 1e-4 --gpu 0
```
- 数据格式：
       - `train/val/test_img_id_gt.txt`：`img_id,text` 每行一条。
       - `chars_list.txt`：每行一个字符。
- 训练前请确保图片已按高 128 等比缩放（预处理脚本会处理）。

## 测试与解码

**单张图片推理：**
```bash
python test.py -m hctr -f <checkpoint> -i <image.png> -dm greedy-search
```

**测试集评估（训练后）：**
```bash
python test.py -m hctr -f <checkpoint> \
    -i data/hwdb2.x \
    -bm \
    -dm greedy-search \
    -b 16 -pf 20
```
- `-bm`：基准测试模式，输入需为数据集目录
- `-dm greedy-search`：默认解码，速度快
- `-dm beam-search`：可叠加语言模型提升精度：
  ```bash
  -dm beam-search --use-tfm-pred --transformer-path <tfm>   # transformer LM
  -dm beam-search --skip-search --kenlm-path <ngram.arpa>   # n-gram LM（低延迟）
  ```
## 部署（可选，OpenVINO）
1) 导出 ONNX：`python utils/export_onnx.py`。
2) 转换为 IR：使用 OpenVINO Model Optimizer。
3) 推理：`python deploy.py -lang hctr -dm greedy-search -i <image>`。

## 许可证
本项目采用 [Apache 2.0 许可证](LICENSE)。

## 引用
如果本项目对你的研究有帮助，请引用：
```
@article{bliu2020hctr-cnn,
       Author = {Brian Liu, Xianchao Xu, Yu Zhang},
       Title = {Offline Handwritten Chinese Text Recognition with Convolutional Neural Networks},
       publisher = {arXiv},
       Year = {2020}
}
```
