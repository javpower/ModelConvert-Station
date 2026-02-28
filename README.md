# ModelConvert-Station

> **URL-Driven Model Conversion Architecture**  
> 将 GitHub Actions 作为无状态云端算力网关，实现纯配置驱动的模型转换流水线。

[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)

---

## 🎯 架构理念

ModelConvert-Station 是一个**配置驱动 (Config-Driven)** 的模型转换架构，核心理念是：

- **零本地依赖**: 所有复杂 Python 环境（PyTorch 2.x, TensorFlow 2.x, Protobuf）只存在于 GitHub Actions 容器中
- **纯 URL 驱动**: 只需提供模型 URL，无需上传二进制文件
- **无状态设计**: 仓库仅存储"转换指令"，体积永远保持在 KB 级别
- **工业级闭环**: 产出不只是模型，而是 **模型 + 结构说明 + Java 代码** 的完整包

---

## 📁 目录结构

```
ModelConvert-Station/
├── .github/workflows/
│   └── convert.yml           # 自动化流水线（监听变更、环境编排、发布成果）
├── engine/                   # 核心转换引擎模块
│   ├── main.py               # 任务调度与异常处理逻辑
│   ├── optimizer.py          # 模型结构压缩与算子优化
│   └── translator/           # 框架专用转换逻辑
│       ├── pytorch_translator.py      # PyTorch -> ONNX
│       ├── tensorflow_translator.py   # TensorFlow/Keras -> ONNX
│       ├── tflite_translator.py       # TFLite -> ONNX
│       └── mediapipe_translator.py    # MediaPipe Task -> ONNX
├── schema/
│   └── task_schema.json      # JSON 校验文件
├── tasks.json                # 用户唯一的交互入口 ⭐
├── requirements.txt          # Python 依赖参考
└── README.md                 # 本文档
```

---

## 🚀 快速开始

### 1. Fork 本仓库

点击右上角 "Fork" 按钮，将仓库复制到你的 GitHub 账户。

### 2. 配置转换任务

编辑 `tasks.json` 文件，添加你的模型 URL：

```json
{
  "tasks": [
    {
      "id": "my_model",
      "source_url": "https://your-domain.com/model.pth",
      "source_framework": "pytorch",
      "custom_args": {
        "input_shape": [1, 3, 224, 224]
      }
    }
  ]
}
```

### 3. 提交触发转换

```bash
git add tasks.json
git commit -m "Add model conversion task"
git push origin main
```

### 4. 获取结果

GitHub Actions 自动执行转换，完成后在 **Releases** 页面下载：
- 转换后的 `.onnx` 模型
- 输入/输出元数据 (`metadata.json`)
- 自动生成的 Java 推理模板 (`Inference.java`)

---

## 📋 tasks.json 配置详解

### 基础配置

```json
{
  "global_config": {
    "opset_version": 17,
    "optimization_level": 2,
    "generate_java_template": true
  },
  "tasks": [
    {
      "id": "unique_task_id",
      "source_url": "https://example.com/model.pth",
      "source_framework": "pytorch",
      "target_format": "onnx",
      "optimization_level": 2,
      "generate_java_template": true,
      "custom_args": {},
      "tags": ["tag1", "tag2"],
      "description": "Task description"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 否 | 任务唯一标识（自动生成） |
| `source_url` | string | **是** | 模型下载 URL（支持 HTTP/HTTPS/Google Drive） |
| `source_framework` | string | 否 | 源框架（自动嗅探） |
| `target_format` | string | 否 | 目标格式（默认 onnx） |
| `optimization_level` | int | 否 | 优化级别 0-3（默认 2） |
| `generate_java_template` | bool | 否 | 生成 Java 模板（默认 true） |
| `custom_args` | object | 否 | 框架特定参数 |
| `tags` | array | 否 | 任务标签 |
| `description` | string | 否 | 任务描述 |

### 框架特定参数

#### PyTorch

```json
{
  "custom_args": {
    "input_shape": [1, 3, 224, 224],
    "input_names": ["input"],
    "output_names": ["output"],
    "opset_version": 17,
    "dynamic_axes": {
      "input": {0: "batch_size"},
      "output": {0: "batch_size"}
    },
    "do_constant_folding": true
  }
}
```

#### TensorFlow / Keras

```json
{
  "custom_args": {
    "input_names": ["input_1"],
    "output_names": ["output_1"],
    "opset_version": 17,
    "inputs_as_nchw": ["input_1"],
    "large_model": false
  }
}
```

#### TensorFlow Lite

```json
{
  "custom_args": {
    "opset_version": 17,
    "dequantize": true,
    "keep_channel_last": false
  }
}
```

#### MediaPipe

```json
{
  "custom_args": {
    "opset_version": 17,
    "convert_all_models": true,
    "dequantize": true
  }
}
```

---

## 🔄 流水线执行流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Trigger   │────▶│  Validation │────▶│   Download  │
│  (Push)     │     │    (JSON)   │     │   (Stream)  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌─────────────┐     ┌────▼────────┐
│   Release   │◀────│   Package   │◀────│  Generate   │
│  (GitHub)   │     │   (ZIP)     │     │    (Java)   │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                         ▲
       │           ┌─────────────┐              │
       └───────────│  Simplify   │◀─────────────┘
                   │   (ONNX)    │
                   └─────────────┘
                          ▲
                          │
                   ┌──────┴──────┐
                   │   Convert   │
                   │  (tf2onnx)  │
                   └─────────────┘
```

---

## 🛠️ 支持的框架

| 框架 | 格式 | 状态 | 说明 |
|------|------|------|------|
| PyTorch | `.pt`, `.pth` | ✅ 完整支持 | 支持完整模型和 state_dict |
| TensorFlow | SavedModel | ✅ 完整支持 | 支持签名自动检测 |
| Keras | `.h5`, `.keras` | ✅ 完整支持 | 支持 Functional 和 Sequential |
| TensorFlow Lite | `.tflite` | ✅ 完整支持 | 支持量化模型反量化 |
| MediaPipe | `.task` | ✅ 完整支持 | 自动解压并转换所有子模型 |
| ONNX | `.onnx` | ✅ 优化支持 | 仅执行简化和优化 |

---

## ⚙️ 优化级别

| 级别 | 名称 | 说明 |
|------|------|------|
| 0 | 无优化 | 仅复制文件 |
| 1 | 基础简化 | 使用 onnx-simplifier 基础模式 |
| 2 | **完整优化**（推荐） | 常量折叠 + 死代码消除 + 形状推断 |
| 3 | 激进优化 | 包含 float64→float32 转换等（可能影响精度） |

---

## 📦 输出结构

每次转换完成后，Release 包包含：

```
convert-20240115_120000.zip
├── {task_id}/
│   ├── {task_id}.onnx              # 转换后的模型
│   ├── {task_id}_optimized.onnx    # 优化后的模型（如启用）
│   ├── metadata.json               # 模型元数据
│   └── Inference.java              # Java 推理模板
├── MANIFEST.json                   # 转换清单
└── conversion_summary.json         # 任务摘要
```

### metadata.json 示例

```json
{
  "task_id": "resnet50",
  "source_url": "https://...",
  "source_framework": "pytorch",
  "conversion_time": "2024-01-15T12:00:00Z",
  "inputs": [
    {
      "name": "input",
      "shape": [1, 3, 224, 224],
      "dtype": "FLOAT"
    }
  ],
  "outputs": [
    {
      "name": "output",
      "shape": [1, 1000],
      "dtype": "FLOAT"
    }
  ],
  "op_types": ["Conv", "Relu", "MaxPool", ...],
  "op_count": 176,
  "file_size_bytes": 102400000
}
```

---

## 🔧 本地开发（高级）

虽然设计为云端执行，你也可以本地运行引擎：

```bash
# 克隆仓库
git clone https://github.com/your-username/ModelConvert-Station.git
cd ModelConvert-Station

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行转换
python engine/main.py --tasks tasks.json --output ./outputs
```

---

## 🎯 特殊模型转换指南

### LOFTR (Local Feature TRansformer)

LOFTR 是双输入的特征匹配模型，支持多输入转换。

**⚠️ 注意**: LOFTR 官方权重托管在 Google Drive，由于访问限制，**建议手动下载后上传到自有存储**：

1. 从 [LoFTR 官方仓库](https://github.com/zju3dv/LoFTR) 下载权重文件 (`indoor_ds.ckpt`, `outdoor_ds.ckpt`)
2. 上传到 Hugging Face Hub / GitHub Release / 自有服务器
3. 更新 `tasks.json` 中的 `source_url`

**LOFTR 配置示例**:

```json
{
  "id": "my_loftr_indoor",
  "source_url": "https://your-domain.com/indoor_ds.ckpt",
  "source_framework": "pytorch",
  "custom_args": {
    "input_shapes": [[1, 1, 480, 640], [1, 1, 480, 640]],
    "input_names": ["image0", "image1"],
    "output_names": ["mkpts0_c", "mkpts1_c", "mconf", "m_bids"],
    "model_architecture": "loftr",
    "dynamic_axes": {
      "image0": {"0": "batch_size", "2": "height", "3": "width"},
      "image1": {"0": "batch_size", "2": "height", "3": "width"}
    }
  }
}
```

**输入说明**:
- 两张灰度图像: `[batch, 1, height, width]`
- 推荐使用尺寸: `480x640` 或 `640x480`
- 图像需要归一化到 `[0, 1]` 范围

**依赖**: 需要安装 `kornia` 库来自动重建模型架构。

---

## 🌐 URL 支持

### HTTP/HTTPS 直链

```json
{
  "source_url": "https://example.com/model.pth"
}
```

### Google Drive

```json
{
  "source_url": "https://drive.google.com/file/d/FILE_ID/view"
}
```

### 支持的存储
- ✅ 直接 HTTP/HTTPS 链接
- ✅ Google Drive 共享链接
- ✅ 支持重定向的短链接
- ✅ 需要确认的大文件（自动处理）

---

## 📝 最佳实践

1. **始终指定 input_shape**: 虽然引擎会尝试推断，但显式指定更可靠
2. **使用有意义的 task ID**: 便于在 Release 中识别
3. **添加 tags 和 description**: 便于任务管理
4. **测试小模型先**: 首次使用建议先用小模型验证流程
5. **合理设置优化级别**: 级别 2 是大多数场景的最佳选择

---

## 🔒 安全说明

- 所有模型下载在临时目录进行，转换后自动清理
- 不存储任何模型二进制到仓库
- 仅通过 GitHub Actions 日志输出转换状态
- 敏感 URL 建议使用 GitHub Secrets（高级用法）

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [ONNX](https://onnx.ai/) - 开放神经网络交换格式
- [tf2onnx](https://github.com/onnx/tensorflow-onnx) - TensorFlow 到 ONNX 转换器
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) - ONNX 模型简化工具

---

<p align="center">
  <strong>ModelConvert-Station</strong> - URL-Driven Model Conversion Architecture
  <br>
  Made with ❤️ for the ML Engineering community
</p>
