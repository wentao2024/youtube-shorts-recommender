# 前端使用说明

## 🚀 快速开始

### 1. 确保后端服务运行

首先启动API服务：
```bash
cd src/api
python3 app.py
```

服务应该在 `http://localhost:8000` 运行

### 2. 打开前端页面

有两种方式：

**方式1：直接打开HTML文件**
- 双击 `index.html` 文件
- 或在浏览器中打开：`file:///path/to/frontend/index.html`

**方式2：使用本地服务器（推荐）**

使用Python启动简单HTTP服务器：
```bash
cd frontend
python3 -m http.server 8080
```

然后在浏览器访问：`http://localhost:8080`

## 📋 功能说明

### 1. 推荐功能
- 输入用户ID和推荐数量
- 可选择自定义召回配置
- 查看推荐结果和分数
- 支持导出推荐结果

### 2. 召回分析
- 查看各路召回的结果
- 了解召回系统的工作过程
- 分析不同召回策略的效果

### 3. 用户统计
- 查看用户的评分历史
- 了解用户活跃度
- 分析用户偏好

### 4. 关于页面
- 系统介绍
- 技术栈说明
- API端点列表

## 🎨 界面特点

- **现代化设计**：使用渐变色彩和卡片式布局
- **响应式布局**：支持桌面和移动设备
- **实时状态**：显示API服务健康状态
- **交互友好**：加载动画、提示消息、错误处理

## ⚙️ 配置

如果需要修改API地址，编辑 `app.js` 文件：

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

改为你的API地址。

## 🐛 常见问题

### CORS错误
如果遇到跨域问题，需要在FastAPI中添加CORS中间件：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 无法连接API
1. 确认API服务正在运行
2. 检查API地址是否正确
3. 查看浏览器控制台的错误信息

## 📱 浏览器支持

- Chrome/Edge (推荐)
- Firefox
- Safari
- 移动浏览器

## 🎯 使用技巧

1. **快速测试**：使用用户ID 1-10 进行测试
2. **查看详情**：点击推荐项查看详细信息
3. **导出数据**：使用导出功能保存推荐结果
4. **监控状态**：关注右上角的服务状态指示器

