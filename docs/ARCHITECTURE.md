# ModelConvert-Station 架构详解

## 设计哲学

ModelConvert-Station 的设计遵循以下核心原则：

### 1. 配置驱动 (Config-Driven)

所有转换行为由 `tasks.json` 单一文件驱动，无需编写代码即可执行复杂转换。

```
用户 ──▶ tasks.json ──▶ GitHub Actions ──▶ ONNX Model + Java Code
```

### 2. 无状态网关 (Stateless Gateway)

GitHub Actions 作为纯粹的无状态算力网关：
- 不存储任何模型二进制
- 每次运行都是全新环境
- 转换结果通过 Release 发布

### 3. 环境解耦 (Environment Decoupling)

| 环境 | 依赖 | 职责 |
|------|------|------|
| 用户本地 | Git | 编辑 tasks.json |
| GitHub Actions | Python + PyTorch + TF + ONNX | 执行转换 |
| 输出产物 | ONNX + Java | 部署推理 |

### 4. 工业级闭环 (Production-Ready)

产出物不仅是模型，而是完整部署包：
- ✅ 优化后的 ONNX 模型
- ✅ 输入/输出元数据（JSON）
- ✅ Java 推理模板（ONNX Runtime）
- ✅ 转换日志和校验报告

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  tasks.json │  │   Git Push  │  │  GitHub Release (Download)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions Layer                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Trigger: Push to tasks.json  │  workflow_dispatch      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │              Validation Stage (Schema Check)            │      │
│  │  • JSON syntax validation                               │      │
│  │  • Schema compliance check                              │      │
│  │  • URL format verification                              │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │              Execution Stage (Ubuntu VM)                │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │      │
│  │  │ Download│─▶│ Detect  │─▶│ Convert │─▶│ Optimize│   │      │
│  │  │ (Stream)│  │Framework│  │(tf2onnx)│  │(onnxsim)│   │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │              Packaging Stage                            │      │
│  │  • Generate Java templates                              │      │
│  │  • Extract metadata                                     │      │
│  │  • Create ZIP archive                                   │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────┐      │
│  │              Release Stage                              │      │
│  │  • Create GitHub Release                                │      │
│  │  • Upload artifacts                                     │      │
│  │  • Generate release notes                               │      │
│  └─────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. StreamDownloader (流式下载器)

```python
class StreamDownloader:
    """
    异步流式下载器
    
    特性:
    - 支持 HTTP/HTTPS 直链
    - 支持 Google Drive 链接（自动处理确认令牌）
    - 支持大文件分块下载
    - 基于 Magic Bytes 的框架嗅探
    """
```

**关键设计**:
- 使用 `aiohttp` 实现异步下载
- 8KB 分块，避免内存溢出
- 自动检测文件类型（不依赖扩展名）

### 2. ConversionEngine (转换引擎)

```python
class ConversionEngine:
    """
    中央任务调度器
    
    职责:
    - 任务队列管理
    - 并发控制（信号量限制）
    - 异常处理与恢复
    - 统计报告生成
    """
```

**关键设计**:
- 信号量控制最大并发数（默认 2）
- 每个任务独立临时目录
- 失败任务不影响其他任务

### 3. Translator 体系 (框架转换器)

```
┌─────────────────────────────────────────┐
│           Base Translator               │
│         (interface definition)          │
└─────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────────┐
│ PyTorch │   │   TF    │   │    TFLite   │
│Translator│   │Translator│   │  Translator │
└─────────┘   └─────────┘   └─────────────┘
                                    │
                                    ▼
                            ┌─────────────┐
                            │ MediaPipe   │
                            │ Translator  │
                            │ (extends    │
                            │  TFLite)    │
                            └─────────────┘
```

### 4. ONNXOptimizer (优化器)

```python
class ONNXOptimizer:
    """
    ONNX 模型优化器
    
    优化级别:
    - Level 0: 无优化
    - Level 1: 基础简化（onnx-simplifier）
    - Level 2: 完整优化（常量折叠 + 死代码消除）
    - Level 3: 激进优化（float64→float32 等）
    """
```

---

## 数据流

### 单次转换的数据流

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  URL    │───▶│ Download│───▶│  Detect │───▶│ Convert │
│ (tasks) │    │ (Stream)│    │Framework│    │(tf2onnx)│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──┴──────┐
│ Release │◀───│ Package │◀───│  Java   │◀───│ Optimize│
│ (ZIP)   │    │ (ZIP)   │    │Template │    │(onnxsim)│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 文件状态转换

```
tasks.json ──▶ [Pending] ──▶ [Downloading] ──▶ [Converting]
                                              │
                                              ▼
[Released] ◀── [Packaging] ◀── [Generating] ◀── [Optimizing]
```

---

## 扩展性设计

### 添加新框架支持

1. 创建 `translator/{framework}_translator.py`
2. 实现 `convert()` 方法
3. 在 `main.py` 中注册

```python
# translator/new_framework_translator.py
class NewFrameworkTranslator:
    async def convert(self, input_path, output_path, custom_args):
        # Implementation
        return {'status': 'success', 'output_path': str(output_path)}

# main.py
self.translators = {
    'new_framework': NewFrameworkTranslator(),
    # ... existing translators
}
```

### 添加新优化器

```python
# optimizer.py
async def custom_optimize(self, model_path, output_path, config):
    """自定义优化策略"""
    model = onnx.load(str(model_path))
    
    # Apply custom optimizations
    # ...
    
    onnx.save(model, str(output_path))
```

---

## 错误处理策略

### 分层错误处理

```
Level 1: Download Error
    └── 重试 3 次 → 失败标记

Level 2: Detection Error
    └── 使用用户指定框架 → 仍失败则标记

Level 3: Conversion Error
    └── 记录详细日志 → 标记失败

Level 4: Optimization Error
    └── 回退到未优化版本 → 警告

Level 5: Packaging Error
    └── 部分打包（有什么包什么）
```

### 失败任务处理

```json
{
  "id": "failed_task",
  "status": "failed",
  "error_message": "Detailed error description",
  "source_url": "https://...",
  "source_framework": "pytorch"
}
```

---

## 性能优化

### 并发控制

```python
# 使用信号量限制并发
semaphore = asyncio.Semaphore(2)

async def process_with_limit(task):
    async with semaphore:
        return await process_task(task)
```

### 缓存策略

| 缓存类型 | 位置 | 有效期 |
|----------|------|--------|
| pip 依赖 | GitHub Actions Cache | 依赖文件哈希变化 |
| 下载文件 | 临时目录（每次运行） | 单次运行 |
| 转换结果 | GitHub Release | 永久 |

### 大文件处理

```python
# 流式下载，避免内存问题
async for chunk in response.content.iter_chunked(8192):
    await f.write(chunk)
```

---

## 安全设计

### 输入安全

- JSON Schema 校验防止注入
- URL 格式验证
- 文件大小限制（GitHub Actions  runner 限制）

### 执行安全

- 临时目录隔离
- 每次运行全新环境
- 无持久化敏感数据

### 输出安全

- 仅通过 Release 发布
- 无第三方服务通信
- 日志脱敏（URL 可能包含 token）

---

## 监控与可观测性

### 日志级别

```
DEBUG: 详细执行步骤
INFO:  关键里程碑
WARNING: 可恢复问题
ERROR: 失败信息
```

### 指标收集

```json
{
  "stats": {
    "total": 10,
    "success": 8,
    "failed": 1,
    "skipped": 1
  }
}
```

---

## 未来扩展

### 计划功能

- [ ] 支持 ONNX 量化（INT8/FP16）
- [ ] 支持 TensorRT 转换
- [ ] 支持 Core ML 转换
- [ ] Web UI 任务管理器
- [ ] 转换历史记录
- [ ] 模型性能基准测试

### 架构演进

```
Current: GitHub Actions (Serverless)
    │
    ▼
Future:  Hybrid (Actions + Self-hosted Runners)
    │
    ▼
Vision:  Kubernetes-native (Argo Workflows)
```

---

## 总结

ModelConvert-Station 的架构优势：

1. **极简交互**: 只需编辑 JSON，无需环境配置
2. **云原生**: 完全基于 GitHub Actions，零运维成本
3. **可扩展**: 模块化设计，易于添加新框架
4. **生产级**: 完整的错误处理、日志、监控
5. **开源友好**: MIT 协议，社区驱动
