# 贡献指南

感谢你对 ModelConvert-Station 的兴趣！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

1. 检查是否已有相关 Issue
2. 创建新 Issue，包含：
   - 问题描述
   - 复现步骤
   - 期望 vs 实际行为
   - 环境信息（如有）

### 提交代码

1. **Fork** 本仓库
2. **创建分支**: `git checkout -b feature/your-feature`
3. **提交更改**: `git commit -m 'Add some feature'`
4. **推送分支**: `git push origin feature/your-feature`
5. **创建 Pull Request**

### 代码规范

#### Python

- 遵循 PEP 8
- 使用类型注解
- 添加 docstring

```python
async def convert(
    self,
    input_path: Path,
    output_path: Path,
    custom_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convert model to ONNX format.
    
    Args:
        input_path: Path to input model
        output_path: Path for output ONNX
        custom_args: Optional configuration
    
    Returns:
        Dictionary with conversion results
    """
```

#### JSON

- 使用 2 空格缩进
- 字段按字母顺序排列

## 开发流程

### 本地开发

```bash
# 克隆仓库
git clone https://github.com/your-username/ModelConvert-Station.git
cd ModelConvert-Station

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/
```

### 添加新框架支持

1. 创建 `translator/{framework}_translator.py`
2. 实现 `convert()` 方法
3. 在 `main.py` 中注册
4. 更新 `task_schema.json`
5. 添加示例配置
6. 更新文档

### 测试

- 添加单元测试
- 确保现有测试通过
- 测试多种模型格式

## 行为准则

- 尊重他人
- 欢迎新手
- 建设性反馈
- 专注于技术

## 许可证

贡献即表示你同意将代码以 MIT 许可证发布。

---

**有疑问？** 开启一个 Discussion 讨论吧！
