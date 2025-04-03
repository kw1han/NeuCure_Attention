# API目录

本目录包含系统对外提供的API接口，用于与其他系统集成或提供远程访问功能。

## 目录结构

- `rest/` - RESTful API接口
  - `server.py` - API服务器
  - `routes/` - API路由
  - `models/` - API数据模型
  - `auth/` - 认证和授权
  
- `websocket/` - WebSocket接口
  - `server.py` - WebSocket服务器
  - `handlers/` - 消息处理器
  
- `sdk/` - 软件开发工具包
  - `python/` - Python SDK
  - `js/` - JavaScript SDK
  
- `docs/` - API文档

## API功能

系统提供以下主要API功能：

1. 会话管理
   - 创建/结束训练会话
   - 查询会话状态和历史

2. 设备控制
   - 连接/断开EEG设备
   - 设备状态监控
   - 校准和测试

3. 数据访问
   - 实时EEG数据流
   - 注意力指标
   - 训练结果和报告

4. 用户管理
   - 用户认证
   - 用户资料管理
   - 权限控制

## API使用示例

### RESTful API

```python
import requests

# 创建会话
response = requests.post(
    "http://localhost:5000/api/sessions",
    json={"user_id": "user123", "session_type": "training"}
)
session_id = response.json()["session_id"]

# 获取注意力数据
response = requests.get(
    f"http://localhost:5000/api/sessions/{session_id}/attention"
)
attention_data = response.json()
```

### WebSocket API

```javascript
// 连接WebSocket
const ws = new WebSocket("ws://localhost:5001/ws/eeg");

// 订阅实时数据
ws.onopen = () => {
    ws.send(JSON.stringify({
        action: "subscribe",
        channel: "attention_data",
        session_id: "session123"
    }));
};

// 接收数据
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("Attention level:", data.attention_level);
};
```

## 安全性

所有API接口都需要认证。系统提供基于JWT (JSON Web Token) 的认证机制。

对于敏感操作（如修改用户数据、访问他人数据），还需要进行适当的授权检查。 