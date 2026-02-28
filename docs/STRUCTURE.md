# ModelConvert-Station 目录结构

```
ModelConvert-Station/
├── .github/
│   └── workflows/
│       └── convert.yml              # GitHub Actions 自动化流水线
│
├── docs/                            # 文档目录
│   ├── ARCHITECTURE.md              # 架构详解
│   ├── FAQ.md                       # 常见问题
│   └── STRUCTURE.md                 # 本文件
│
├── engine/                          # 核心转换引擎
│   ├── main.py                      # 任务调度与异常处理
│   ├── optimizer.py                 # ONNX 优化器
│   └── translator/                  # 框架转换器集合
│       ├── __init__.py
│       ├── pytorch_translator.py    # PyTorch -> ONNX
│       ├── tensorflow_translator.py # TensorFlow/Keras -> ONNX
│       ├── tflite_translator.py     # TFLite -> ONNX
│       └── mediapipe_translator.py  # MediaPipe Task -> ONNX
│
├── examples/                        # 示例配置
│   ├── pytorch_resnet50.json        # PyTorch ResNet50
│   ├── tensorflow_mobilenet.json    # TensorFlow MobileNet
│   ├── tflite_quantized.json        # TFLite 量化模型
│   ├── mediapipe_tasks.json         # MediaPipe 任务
│   └── advanced_batch.json          # 高级批量转换
│
├── schema/
│   └── task_schema.json             # JSON Schema 校验定义
│
├── .gitignore                       # Git 忽略规则
├── CONTRIBUTING.md                  # 贡献指南
├── LICENSE                          # MIT 许可证
├── README.md                        # 项目主文档
├── requirements.txt                 # Python 依赖参考
└── tasks.json                       # 用户任务配置入口 ⭐
```

## 文件说明

### 核心文件

| 文件 | 用途 | 用户编辑 |
|------|------|----------|
| `tasks.json` | 转换任务配置 | ✅ 是 |
| `.github/workflows/convert.yml` | CI/CD 流水线 | ❌ 否 |
| `engine/main.py` | 转换引擎入口 | ❌ 否 |
| `schema/task_schema.json` | 配置校验规则 | ❌ 否 |

### 文档文件

| 文件 | 内容 |
|------|------|
| `README.md` | 项目介绍、快速开始 |
| `docs/ARCHITECTURE.md` | 系统架构详解 |
| `docs/FAQ.md` | 常见问题解答 |
| `CONTRIBUTING.md` | 贡献指南 |

### 示例文件

| 文件 | 说明 |
|------|------|
| `examples/pytorch_resnet50.json` | PyTorch 图像分类模型 |
| `examples/tensorflow_mobilenet.json` | TensorFlow 轻量模型 |
| `examples/tflite_quantized.json` | 边缘量化模型 |
| `examples/mediapipe_tasks.json` | 视觉任务模型 |
| `examples/advanced_batch.json` | 批量转换高级配置 |

## 关键路径

```
用户输入 ──▶ tasks.json
                 │
                 ▼
           GitHub Actions
                 │
                 ▼
           engine/main.py
                 │
                 ▼
           translator/*.py
                 │
                 ▼
           optimizer.py
                 │
                 ▼
           GitHub Release
```

## 扩展路径

添加新框架支持：

```
1. 创建 engine/translator/{framework}_translator.py
2. 在 engine/main.py 中注册
3. 更新 schema/task_schema.json
4. 添加 examples/{framework}_example.json
5. 更新文档
```
