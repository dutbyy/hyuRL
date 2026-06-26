# hyurl

一个轻量级的强化学习库，旨在提供灵活、高效的强化学习算法实现。

## 项目简介

hyurl 是一个轻量级的强化学习库，旨在提供灵活、高效的强化学习算法实现。它支持多种网络架构和算法，适用于各种强化学习任务。该库采用模块化设计，用户可以轻松地自定义网络架构、特征提取和训练策略。

### 主要特性

- **模块化设计**：网络架构、特征提取、算法实现等模块高度解耦，便于自定义和扩展
- **灵活的网络架构**：支持多种编码器、解码器和聚合器，可以灵活组合构建复杂的神经网络
- **分布式训练**：支持 gRPC 通信，便于实现分布式训练和推理
- **丰富的算法支持**：目前支持 PPO 等主流强化学习算法
- **完善的工具集**：提供日志记录、性能监控、数据可视化等工具

## 项目结构

```
hyurl/
├── hyurl/           # 主源代码
│   ├── algo/        # 算法实现
│   │   ├── PPOPolicy.py
│   │   └── PPOPolicyMinibatch.py
│   ├── api/         # API 定义
│   │   └── net/     # 网络配置 API
│   ├── expert/      # 专家控制器
│   ├── feature/     # 特征提取
│   │   ├── feature.py
│   │   └── feature_set.py
│   ├── flow/        # 流程相关代码
│   │   └── drill_plugin/  # Drill 插件
│   ├── loss/        # 损失函数
│   ├── memory/      # 记忆/缓冲区实现
│   │   └── buffer.py
│   ├── network/     # 网络架构
│   │   ├── encoder/ # 编码器
│   │   ├── decoder/ # 解码器
│   │   ├── aggregator/ # 聚合器
│   │   ├── complex.py
│   │   ├── app_value.py
│   │   └── commander.py
│   └── tools/       # 工具函数
│       ├── common.py
│       ├── gpu.py
│       └── infer.py
├── example/         # 示例实现
│   ├── cartpole/    # CartPole 示例
│   ├── atari/       # Atari 示例
│   ├── acrobot/     # Acrobot 示例
│   ├── montaincar/  # MountainCar 示例
│   └── minigame/    # MiniGame 示例
├── tests/           # 测试文件
│   ├── native/      # 本地测试
│   │   ├── predictor.py
│   │   ├── sampler.py
│   │   ├── trainer.py
│   │   └── eval.py
│   ├── flow/        # 流程测试
│   └── proto/       # Protocol Buffer 定义
├── setup.py         # 安装配置
└── pyproject.toml   # 项目配置
```

## 安装

### 使用 uv 安装（推荐）

```bash
# 克隆项目
git clone https://github.com/yourusername/hyurl.git
cd hyurl

# 使用 uv 同步依赖
uv sync
```

### 使用 pip 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/hyurl.git
cd hyurl

# 安装项目
pip install -e .
```

## 依赖项

- Python >= 3.6
- torch
- grpcio
- protobuf
- numpy
- networkx
- tensorboard
- gym (可选，用于示例环境)

## 编译 Proto 文件

如果需要修改 Protocol Buffer 定义，可以使用以下命令重新编译：

```bash
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. tests/proto/predictor.proto
```

## 快速开始

### 运行测试

#### 1. 启动 gRPC 服务器

```bash
uv run python tests/native/predictor.py
```

服务器将在 `localhost:50051` 上启动，等待客户端连接。

#### 2. 运行测试客户端

```bash
uv run python tests/native/test.py
```

测试客户端将向服务器发送推理请求，验证服务器的功能。

### 运行训练

#### 1. 启动 gRPC 服务器

```bash
uv run python tests/native/predictor.py
```

#### 2. 启动训练

```bash
uv run python tests/native/eval.py
```

训练过程将启动多个采样进程，收集数据并更新模型权重。

## 使用示例

### 基本网络配置

```python
from hyurl.network import *
from hyurl.feature.feature import *
from hyurl.feature.feature_set import *
from hyurl.api.net.net import *

# 定义网络配置
network_cfg = {
    "encoder_demo": {
        "class": CommonEncoder,
        "params": {
            "in_features": 4,
            "hidden_layer_sizes": [256, 128],
        },
        "inputs": ['feature_a']
    },
    "aggregator": {
        "class": DenseAggregator,
        "params": {
            "in_features": 128,
            "hidden_layer_sizes": [256, 128],
            "output_size": 256
        },
        "inputs": ['encoder_demo']
    },
    "value_app": {
        "class": ValueApproximator,
        "params": {
            "in_features": 256,
            "hidden_layer_sizes": [256, 256, 128],
        },
        "inputs": ['aggregator']
    },
    "action": {
        "class": CategoricalDecoder,
        "params": {
            "n": 2,
            "hidden_layer_sizes": [128, 128],
        },
        "inputs": ['aggregator']
    }
}

# 创建网络
from hyurl.network import ComplexNetwork
model = ComplexNetwork(network_cfg)

# 进行推理
input_data = {"feature_a": torch.rand(1, 4)}
output = model(input_data)
print(output)
```

### 使用 PPO 算法训练

```python
from hyurl.algo.PPOPolicy import PPOPolicy
from hyurl.memory.buffer import Memory

# 定义策略配置
policy_config = {
    "network_cfg": network_cfg,
    "learning_rate": 3e-4,
    "clip_ratio": 0.2,
    "epochs": 10,
    "batch_size": 64
}

# 创建策略
policy = PPOPolicy(policy_config=policy_config, device='cpu')

# 创建缓冲区
buffer = Memory()

# 训练循环
for episode in range(1000):
    # 采样数据
    state = env.reset()
    for step in range(1000):
        # 获取动作
        action = policy.act(state)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 存储数据
        buffer.store(state, action, reward, done)
        
        if done:
            break
        
        state = next_state
    
    # 训练策略
    train_data = buffer.get_batch(batch_size=64)
    policy.learn(train_data)
```

### 使用 gRPC 服务器

```python
from tests.native.predictor import PredictorClient

# 创建客户端
client = PredictorClient('localhost', 50051)

# 进行推理
state_dict = {
    "model": "model_name",
    "obs": {"feature_a": np.random.rand(4)}
}

outputs, err_code = await client.predict(state_dict)
print(outputs)
```

## 支持的算法

### PPO (Proximal Policy Optimization)

PPO 是一种近端策略优化算法，通过限制策略更新的幅度来保证训练的稳定性。

- **PPOPolicy**: 标准 PPO 实现
- **PPOPolicyMinibatch**: 支持小批量训练的 PPO 实现

## 网络架构

### 编码器 (Encoders)

- **CommonEncoder**: 通用编码器，适用于向量特征
- **EntityEncoder**: 实体编码器，适用于实体特征
- **SpatialEncoder**: 空间编码器，适用于空间特征

### 解码器 (Decoders)

- **CategoricalDecoder**: 分类解码器，适用于离散动作空间
- **GaussianDecoder**: 高斯解码器，适用于连续动作空间
- **SingleSelectiveDecoder**: 单选解码器，适用于单选动作
- **OrderedMultiSelectiveDecoder**: 有序多选解码器，适用于有序多选动作
- **UnorderedMultiSelectiveDecoder**: 无序多选解码器，适用于无序多选动作

### 聚合器 (Aggregators)

- **DenseAggregator**: 密集聚合器，使用全连接层聚合编码器输出

### 其他组件

- **ValueApproximator**: 值函数近似器，用于估计状态值
- **ComplexNetwork**: 复杂网络，支持有向无环图结构的网络构建

## 特征工程

### 特征类型

- **VectorFeature**: 向量特征
- **OnehotFeature**: 独热编码特征

### 特征集

- **CommonFeatureSet**: 通用特征集
- **EntityFeatureSet**: 实体特征集
- **SpatialFeatureSet**: 空间特征集

## 工具函数

### 日志记录

```python
from hyurl.tools.common import Summary

# 记录标量
Summary.add_scalar("loss", 0.5, global_step=100)

# 设置日志路径
Summary.setpath("experiment_1")
```

### 性能监控

```python
from hyurl.tools.common import timer_decorator

@timer_decorator
def train_step():
    # 训练代码
    pass
```

### GPU 工具

```python
from hyurl.tools.gpu import auto_move

# 自动将数据移动到 GPU
auto_move(data, 'cuda')
```

## 配置说明

### 网络配置

网络配置使用字典格式，包含以下字段：

- `class`: 网络组件类
- `params`: 网络组件参数
- `inputs`: 输入来源

示例：

```python
{
    "encoder_demo": {
        "class": CommonEncoder,
        "params": {
            "in_features": 4,
            "hidden_layer_sizes": [256, 128],
        },
        "inputs": ['feature_a']
    }
}
```

### 策略配置

策略配置包含网络配置和训练参数：

```python
{
    "network_cfg": network_cfg,
    "learning_rate": 3e-4,
    "clip_ratio": 0.2,
    "epochs": 10,
    "batch_size": 64
}
```

## 示例项目

### CartPole

```bash
cd example/cartpole
uv run python local.py
```

### Atari

```bash
cd example/atari
uv run python local.py
```

### Acrobot

```bash
cd example/acrobot
uv run python local.py
```

## 贡献

欢迎提交 Issue 和 Pull Request！

### 开发指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加必要的注释和文档字符串
- 编写单元测试

## 许可证

MIT License

## 联系方式

- 项目主页: https://github.com/yourusername/hyurl
- 问题反馈: https://github.com/yourusername/hyurl/issues

## 致谢

感谢所有为本项目做出贡献的开发者。
