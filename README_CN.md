# 音频转换器

[English](README.md) | [中文](README_CN.md)

一个将音频转换为离散表示并使用Transformer模型进行处理的音频处理流水线。

## 处理流程

1. 音频处理:
   - 将多通道音频转换为单通道
   - 分帧和FFT变换
   - 提取离散频谱
   - Mel滤波
   - 离散梅尔频谱（频域离散化）

2. 可选矢量量化:
   - VQ-VAE/聚类方法生成更紧凑的离散词元

3. Transformer模型:
   - 序列长度 = 帧数
   - 词元维度 = 特征维数

## 架构详情

### 音频处理
- `AudioProcessor`类负责音频加载、重采样和特征提取
- 音频被转换为单通道并处理成梅尔频谱图
- 默认参数：22050Hz采样率、1024 FFT大小、512跳跃长度、80个梅尔频带

### 矢量量化 (VQ-VAE)
- 将连续的梅尔频谱图压缩为离散词元
- 由编码器、矢量量化器和解码器组成
- 编码器：使用BatchNorm和ReLU激活函数的Conv1D层
- 矢量量化器：将连续向量映射到学习到的码本中最近的条目
- 解码器：从量化表示中重建梅尔频谱图

### Transformer模型
- 两种实现:
  1. `AudioTransformer`: 处理来自VQ-VAE的离散词元
  2. `AudioFeatureTransformer`: 直接处理连续特征

- 标准架构使用:
  - 512维嵌入
  - 8个注意力头
  - 6个编码器层
  - 6个解码器层（完整Transformer）
  - 位置编码用于序列感知

## 安装

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/audio-conv-transformer.git
cd audio-conv-transformer
```

2. 安装所需的包:
```bash
pip install -r requirements.txt
```

## 使用方法

### 快速演示
运行演示脚本来测试音频处理和可视化:
```bash
python demo.py --audio_file 音频文件路径.wav
```

### 训练
训练VQ-VAE模型:
```bash
python run.py train --audio_dir 音频文件目录 --model_type vqvae --batch_size 16 --num_epochs 100
```

训练Transformer模型（在训练VQ-VAE之后）:
```bash
python run.py train --audio_dir 音频文件目录 --model_type transformer --vqvae_checkpoint checkpoints/vqvae_epoch_100.pt
```

训练特征Transformer（不使用VQ-VAE）:
```bash
python run.py train --audio_dir 音频文件目录 --model_type feature_transformer
```

### 推理
处理音频文件并可视化结果:
```bash
python run.py inference --vqvae_checkpoint checkpoints/vqvae_epoch_100.pt --audio_file 音频文件路径.wav
```

使用transformer模型生成音频:
```bash
python run.py inference --vqvae_checkpoint checkpoints/vqvae_epoch_100.pt --transformer_checkpoint checkpoints/transformer_epoch_100.pt --generate
```

## 项目结构

```
audio_transformer/
├── __init__.py          # 包初始化
├── data/                # 数据处理脚本和数据集
├── models/              # 神经网络模型
│   ├── __init__.py      # 模型包初始化
│   ├── transformer.py   # Transformer模型实现
│   └── vq_vae.py        # VQ-VAE模型实现
└── utils/               # 实用工具函数
    ├── __init__.py      # 工具包初始化
    └── audio_processing.py  # 音频处理工具
demo.py                  # 演示脚本，用于可视化
requirements.txt         # 包依赖
run.py                   # 主运行脚本
```

## 高级配置

您可以自定义各种参数:

- 音频处理参数:
  - `--sample_rate`: 音频采样率（默认: 22050）
  - `--n_fft`: FFT大小（默认: 1024）
  - `--hop_length`: 跳跃长度（默认: 512）
  - `--n_mels`: 梅尔频带数量（默认: 80）

- 训练参数:
  - `--learning_rate`: 学习率（默认: 1e-4）
  - `--batch_size`: 批量大小（默认: 16）
  - `--num_epochs`: 训练轮数（默认: 100）
  - `--val_split`: 验证集比例（默认: 0.1）

- 生成参数:
  - `--max_length`: 生成的最大序列长度（默认: 1000）
  - `--temperature`: 采样温度（默认: 1.0）

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- torchaudio 0.9.0+
- librosa 0.8.1+
- datasets 2.10.0+
- transformers 4.26.0+
- 其他依赖项请查看requirements.txt 

## 默认数据集

为方便起见，我们推荐以下数据集用于训练和评估：

1. **GTZAN流派集合**：
   - 1000个音频片段（每个30秒）
   - 10种流派（蓝调、古典、乡村、迪斯科、嘻哈、爵士、金属、流行、雷鬼、摇滚）
   - 大小：约1.2GB
   - [下载链接](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

2. **免费口语数字数据集 (FSDD)**：
   - 适合测试的小型数据集
   - 口语数字（0-9）有多个说话者
   - 大小：约10MB
   - [GitHub仓库](https://github.com/Jakobovski/free-spoken-digit-dataset)

3. **UrbanSound8K**：
   - 8732个标记的城市声音片段（≤4秒）
   - 10个类别
   - 大小：约6GB
   - [下载链接](https://urbansounddataset.weebly.com/urbansound8k.html)

使用这些数据集进行项目训练：

```bash
# 对于GTZAN
python run.py train --audio_dir GTZAN数据路径/genres --model_type vqvae --dataset_type gtzan

# 对于FSDD
python run.py train --audio_dir FSDD数据路径/recordings --model_type vqvae --dataset_type fsdd

# 对于UrbanSound8K
python run.py train --audio_dir UrbanSound8K数据路径/audio --model_type vqvae --dataset_type urbansound
```

项目包含这些常用数据集的内置加载器，可以处理特定的目录结构和元数据格式。 