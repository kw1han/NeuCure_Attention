# 基于非侵入式脑机接口的儿童注意力康复训练系统

该系统融合脑机接口技术与趣味互动设计，旨在帮助儿童在游戏中自然提升注意力。通过非侵入式脑机接口实时监测、互动
任务激发兴趣，以及个性化训练方案的精准匹配，显著提高注意力训练的效率与家庭可执行性，帮助家长和教育者获得量化可视化的干预依据。
                                             
## 核心功能

### 1. 实时注意力监测
通过头戴式非侵入式脑机接口设备（8通道OpenBCI），采集儿童的脑电波（EEG）信号，实时分析其注意力状态，并进行数
据可视化呈现。系统同时支持HBN EEG数据集用于研究和训练。                                                                                              
### 2. 互动式注意力训练任务
将检测到的注意力状态转化为互动游戏训练场景：
- 太空宝贝：通过注意力控制飞船的速度和避障能力
- 魔法森林大冒险：根据注意力水平调整干扰选项的数量
- 色彩拼图奇遇：注意力水平决定拼图块的旋转速度和响应时间

### 3. 个性化训练方案
系统根据实时脑电数据和训练表现，自动调整训练难度和类型，并生成个性化训练报告，帮助家长和康复师更精细地跟踪与
优化训练过程。                                                                                              
### 4. 便携式设计与家长端支持
设备轻便易佩戴，适配家庭和学校场景；配套移动端应用便于家长随时查看训练进度、效果报告，远程管理孩子的训练计划
。                                                                                                          
## 技术实现

- 脑电信号采集：使用8通道OpenBCI设备，采样率为256Hz
- 信号处理：Daubechies 4小波变换进行多层分解，提取功率谱密度（PSD）等特征值
- 机器学习算法：采用随机森林算法对提取的特征数据进行分类，评估注意力水平
- 闭环生物反馈：将注意力状态实时映射到游戏任务中，实现互动式训练
- 个性化推荐：基于历史训练数据和实时脑电信号动态调整训练难度
- API服务：提供RESTful和WebSocket API用于数据访问和实时注意力监控

## 安装与使用

### 系统要求
- Python 3.8+
- OpenBCI硬件设备（可选，支持模拟数据和HBN数据集）
- 支持的操作系统：Windows 10+, macOS 10.14+, Linux

### 安装步骤
1. 克隆代码库：
```
git clone https://github.com/yourusername/brain_attention_system.git
```

2. 创建虚拟环境并安装依赖：
```
cd brain_attention_system
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. 运行系统：
```
python main.py
```

### 使用HBN EEG数据

本系统支持使用HBN（Healthy Brain Network）EEG数据集进行注意力分析和训练。

#### 数据获取与准备

1. 从[HBN官方网站](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/sharing_neuro.html)下载EEG数据
2. 将下载的数据解压到项目的`data/eeg/hbn`目录下
3. 系统会自动识别和处理这些数据文件

#### HBN数据应用场景

- 科研分析：使用真实儿童EEG数据进行算法开发和验证
- 模型训练：基于大量样本数据训练更精确的注意力分类模型
- 离线测试：在没有实时硬件设备的情况下进行系统测试

### 启动API服务器

系统提供了完整的API接口，可以通过以下命令启动：

```bash
# 启动所有API服务（RESTful和WebSocket）
python api_server.py

# 仅启动REST API服务器
python api_server.py --only-rest

# 仅启动WebSocket服务器
python api_server.py --only-ws

# 指定HBN数据目录
python api_server.py --data-dir path/to/your/hbn/data
```

API服务器默认端口：
- RESTful API: 5000
- WebSocket API: 5001

详细的API文档和使用示例请参见[API文档](api/README.md)。

## 项目结构
```
brain_attention_system/
├── api/                 # API接口
│   ├── rest/            # RESTful API
│   ├── websocket/       # WebSocket API
│   └── client_example.py # API客户端示例
├── assets/              # 图像、音频资源
├── components/          # UI组件
├── config/              # 配置文件
├── data/                # 数据存储
│   ├── eeg/             # EEG数据
│   │   ├── hbn/         # HBN数据集
│   │   └── recorded/    # 实时记录的数据
│   ├── reports/         # 训练报告
│   └── user_profiles/   # 用户档案
├── models/              # 机器学习模型
├── utils/               # 工具函数
│   └── hbn_eeg_loader.py # HBN数据加载器
├── main.py              # 主程序入口
├── api_server.py        # API服务器入口
└── requirements.txt     # 项目依赖
```

## 许可证
MIT 