# 快速开始指南

## 5 分钟上手 ModelConvert-Station

### 第 1 步：Fork 仓库

点击 GitHub 页面右上角的 **Fork** 按钮，将仓库复制到你的账户。

### 第 2 步：编辑任务配置

打开 `tasks.json` 文件，编辑为你的模型：

```json
{
  "tasks": [
    {
      "id": "my_first_model",
      "source_url": "https://your-domain.com/model.pth",
      "source_framework": "pytorch",
      "custom_args": {
        "input_shape": [1, 3, 224, 224]
      }
    }
  ]
}
```

### 第 3 步：提交更改

```bash
git add tasks.json
git commit -m "Add my model conversion task"
git push origin main
```

或者直接在线编辑并提交。

### 第 4 步：查看转换结果

1. 点击仓库的 **Actions** 标签
2. 等待工作流完成（通常 2-5 分钟）
3. 点击 **Releases** 标签下载结果

## 常见模型配置

### PyTorch 图像分类

```json
{
  "id": "pytorch_classifier",
  "source_url": "https://example.com/model.pth",
  "source_framework": "pytorch",
  "custom_args": {
    "input_shape": [1, 3, 224, 224],
    "input_names": ["input"],
    "output_names": ["output"]
  }
}
```

### TensorFlow SavedModel

```json
{
  "id": "tf_model",
  "source_url": "https://example.com/saved_model.zip",
  "source_framework": "tensorflow"
}
```

### TFLite 量化模型

```json
{
  "id": "tflite_quantized",
  "source_url": "https://example.com/model.tflite",
  "source_framework": "tflite",
  "custom_args": {
    "dequantize": true
  }
}
```

### MediaPipe 任务

```json
{
  "id": "mediapipe_pose",
  "source_url": "https://example.com/pose_landmarker.task",
  "source_framework": "mediapipe"
}
```

## 故障排除

### 转换失败

1. 查看 **Actions** 页面的日志
2. 确认 URL 可访问
3. 检查 `input_shape` 是否正确

### 找不到输出

1. 确保工作流成功完成（绿色 ✓）
2. 检查 **Releases** 页面
3. 查看工作流日志中的 artifact 上传信息

## 下一步

- 阅读 [完整文档](README.md)
- 了解 [架构设计](docs/ARCHITECTURE.md)
- 查看 [常见问题](docs/FAQ.md)
- 浏览 [更多示例](examples/)

---

**🎉 恭喜！你已完成第一次模型转换。**
