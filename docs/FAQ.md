# ModelConvert-Station 常见问题

## 基础问题

### Q: 什么是 ModelConvert-Station？

**A:** ModelConvert-Station 是一个 URL 驱动的模型转换架构，利用 GitHub Actions 作为无状态云端算力网关，将各种深度学习框架（PyTorch、TensorFlow、TFLite、MediaPipe）的模型自动转换为 ONNX 格式。

### Q: 为什么使用 GitHub Actions？

**A:** 
- **零成本**: 公共仓库免费使用
- **零配置**: 无需管理服务器
- **预装环境**: Ubuntu + Python 生态开箱即用
- **天然集成**: 与 Git 工作流无缝结合

### Q: 支持哪些框架？

**A:**
| 框架 | 格式 | 状态 |
|------|------|------|
| PyTorch | `.pt`, `.pth` | ✅ 完整支持 |
| TensorFlow | SavedModel | ✅ 完整支持 |
| Keras | `.h5`, `.keras` | ✅ 完整支持 |
| TensorFlow Lite | `.tflite` | ✅ 完整支持 |
| MediaPipe | `.task` | ✅ 完整支持 |
| ONNX | `.onnx` | ✅ 优化支持 |

---

## 配置问题

### Q: 如何找到模型的 input_shape？

**A:** 几种方法：

1. **查看模型文档**: 官方通常会提供输入尺寸
2. **使用 Netron**: 可视化模型查看输入节点
3. **框架代码**: 查看模型定义中的第一层
4. **试错法**: 从常见尺寸开始尝试（[1, 3, 224, 224]）

### Q: 支持动态 batch size 吗？

**A:** 支持！使用 `dynamic_axes` 配置：

```json
{
  "custom_args": {
    "input_shape": [1, 3, 224, 224],
    "dynamic_axes": {
      "input": {0: "batch_size"},
      "output": {0: "batch_size"}
    }
  }
}
```

### Q: 如何转换 Google Drive 上的模型？

**A:** 直接使用分享链接：

```json
{
  "source_url": "https://drive.google.com/file/d/FILE_ID/view"
}
```

系统会自动处理下载确认令牌。

### Q: 可以一次转换多个模型吗？

**A:** 可以！在 `tasks` 数组中添加多个任务：

```json
{
  "tasks": [
    {"id": "model1", "source_url": "..."},
    {"id": "model2", "source_url": "..."},
    {"id": "model3", "source_url": "..."}
  ]
}
```

---

## 转换问题

### Q: 转换失败了怎么办？

**A:** 

1. **查看 Actions 日志**: 点击仓库的 Actions 标签查看详细错误
2. **检查 URL**: 确保链接可访问且不需要登录
3. **验证 input_shape**: 确保输入形状与模型匹配
4. **尝试降低优化级别**: 设置 `optimization_level: 1`

### Q: 为什么 MediaPipe 转换出多个 ONNX 文件？

**A:** MediaPipe `.task` 文件是 ZIP 压缩包，内部包含多个 TFLite 模型（如检测器 + 关键点回归器）。系统会自动解压并转换所有子模型。

### Q: 量化模型支持吗？

**A:** 支持！TFLite 量化模型会自动反量化：

```json
{
  "custom_args": {
    "dequantize": true
  }
}
```

### Q: 转换后的模型精度会变化吗？

**A:** 
- **优化级别 0-2**: 精度保持不变
- **优化级别 3**: 可能将 float64 转为 float32，理论上精度损失极小

---

## 输出问题

### Q: 输出文件在哪里？

**A:** 在 GitHub 仓库的 **Releases** 页面，每次转换会创建一个新的 Release。

### Q: Java 模板有什么用？

**A:** Java 模板是基于 ONNX Runtime 的完整推理代码，包含：
- 模型加载
- 输入/输出处理
- 推理执行
- 资源释放

可以直接集成到 Android 或 Java 后端项目中。

### Q: 如何禁用 Java 模板生成？

**A:** 

```json
{
  "generate_java_template": false
}
```

### Q: metadata.json 包含什么？

**A:** 包含模型的完整元数据：
- 输入/输出节点名称和形状
- 数据类型
- 算子类型列表
- 转换时间戳
- 源框架信息

---

## 性能问题

### Q: 转换需要多长时间？

**A:** 取决于模型大小：
- 小型模型 (< 10MB): 1-2 分钟
- 中型模型 (10-100MB): 3-5 分钟
- 大型模型 (> 100MB): 5-15 分钟

### Q: 有并发限制吗？

**A:** 默认最多同时转换 2 个模型，避免资源耗尽。

### Q: 可以加速转换吗？

**A:** 
- 使用 GitHub Actions 的 `workflow_dispatch` 触发特定任务
- 减少优化级别
- 确保 URL 下载速度快

---

## 高级问题

### Q: 可以本地运行吗？

**A:** 可以！

```bash
pip install -r requirements.txt
python engine/main.py --tasks tasks.json --output ./outputs
```

### Q: 如何添加自定义转换器？

**A:** 参考 `docs/ARCHITECTURE.md` 的扩展性设计章节。

### Q: 支持私有模型吗？

**A:** 支持！几种方式：
1. 使用带 token 的 URL（注意安全性）
2. 使用 GitHub Secrets（需要修改 workflow）
3. 本地运行

### Q: 可以转换到 TensorRT 吗？

**A:** 目前不支持，但计划在未来版本添加。

---

## 故障排除

### 错误: "Cannot detect framework"

**解决:** 显式指定 `source_framework`：

```json
{
  "source_framework": "pytorch"
}
```

### 错误: "Input shape not provided"

**解决:** 在 `custom_args` 中添加 `input_shape`：

```json
{
  "custom_args": {
    "input_shape": [1, 3, 224, 224]
  }
}
```

### 错误: "Download failed"

**解决:**
- 检查 URL 是否可访问
- 确保不需要登录
- 对于 Google Drive，确保文件是公开分享的

### 错误: "ONNX validation failed"

**解决:**
- 尝试降低优化级别
- 检查模型是否包含自定义算子
- 使用 Netron 检查模型结构

---

## 其他问题

### Q: 这个项目是免费的吗？

**A:** 是的！MIT 开源协议，完全免费。

### Q: 如何贡献代码？

**A:** 参考 `CONTRIBUTING.md` 文件。

### Q: 发现 Bug 怎么办？

**A:** 在 GitHub Issues 中提交，包含：
- 任务配置（脱敏后）
- 错误日志
- 期望行为

### Q: 有讨论社区吗？

**A:** 使用 GitHub Discussions 进行交流。

---

**还有问题？** 提交一个 [GitHub Issue](https://github.com/your-username/ModelConvert-Station/issues/new) 吧！
