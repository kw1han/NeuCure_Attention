#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API客户端示例
展示如何使用REST和WebSocket API
"""

import os
import sys
import json
import time
import asyncio
import requests
import websockets
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List

# 配置
REST_API_URL = "http://localhost:5000"
WEBSOCKET_API_URL = "ws://localhost:5001"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

class BrainAttentionAPIClient:
    """脑注意力系统API客户端"""
    
    def __init__(self, rest_url: str = REST_API_URL, ws_url: str = WEBSOCKET_API_URL):
        """初始化客户端"""
        self.rest_url = rest_url
        self.ws_url = ws_url
        self.token = None
        self.client_id = None
        self.websocket = None
        self.last_attention_scores = []  # 存储最近的注意力分数，用于绘图
        self.subjects = []  # 存储可用的被试ID
        
    def login(self, username: str, password: str) -> bool:
        """登录并获取访问令牌"""
        url = f"{self.rest_url}/api/auth/login"
        data = {
            "username": username,
            "password": password
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                self.token = result.get("token")
                print(f"登录成功: {result.get('user_id')}")
                print(f"令牌有效期至: {result.get('expires_at')}")
                return True
            else:
                print(f"登录失败: {response.json().get('error')}")
                return False
        except Exception as e:
            print(f"登录时出错: {str(e)}")
            return False
    
    def get_auth_header(self) -> Dict[str, str]:
        """获取包含认证令牌的请求头"""
        if not self.token:
            raise ValueError("未登录，请先调用login方法")
        
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def get_subjects(self) -> List[str]:
        """获取所有可用的被试ID"""
        url = f"{self.rest_url}/api/subjects"
        
        try:
            response = requests.get(url, headers=self.get_auth_header())
            if response.status_code == 200:
                result = response.json()
                self.subjects = result.get("subjects", [])
                print(f"找到 {len(self.subjects)} 个被试:")
                for subject in self.subjects:
                    print(f"  - {subject}")
                return self.subjects
            else:
                print(f"获取被试列表失败: {response.json().get('error')}")
                return []
        except Exception as e:
            print(f"获取被试列表时出错: {str(e)}")
            return []
    
    def get_subject_data(self, subject_id: str, preprocess: bool = True) -> Optional[Dict[str, Any]]:
        """获取特定被试的数据"""
        url = f"{self.rest_url}/api/subjects/{subject_id}"
        params = {"preprocess": str(preprocess).lower()}
        
        try:
            response = requests.get(url, headers=self.get_auth_header(), params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取被试数据失败: {response.json().get('error')}")
                return None
        except Exception as e:
            print(f"获取被试数据时出错: {str(e)}")
            return None
    
    def get_subject_raw_data(self, subject_id: str, start: float = 0, duration: float = 10, 
                           preprocess: bool = True) -> Optional[Dict[str, Any]]:
        """获取特定被试的原始EEG数据"""
        url = f"{self.rest_url}/api/subjects/{subject_id}/raw"
        params = {
            "preprocess": str(preprocess).lower(),
            "start": start,
            "duration": duration
        }
        
        try:
            response = requests.get(url, headers=self.get_auth_header(), params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取原始数据失败: {response.json().get('error')}")
                return None
        except Exception as e:
            print(f"获取原始数据时出错: {str(e)}")
            return None
    
    def simulate_eeg_data(self, seconds: int = 60, sample_rate: int = 256, 
                        attention_level: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """生成模拟的EEG数据"""
        url = f"{self.rest_url}/api/simulate"
        data = {
            "seconds": seconds,
            "sample_rate": sample_rate,
            "attention_level": attention_level
        }
        
        try:
            response = requests.post(url, headers=self.get_auth_header(), json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"模拟数据失败: {response.json().get('error')}")
                return None
        except Exception as e:
            print(f"模拟数据时出错: {str(e)}")
            return None
            
    def predict_attention(self, eeg_data: List[List[float]], channels: List[str], 
                        sample_rate: float) -> Optional[Dict[str, Any]]:
        """从原始EEG数据预测注意力水平"""
        url = f"{self.rest_url}/api/attention/predict"
        data = {
            "eeg_data": eeg_data,
            "channels": channels,
            "sample_rate": sample_rate
        }
        
        try:
            response = requests.post(url, headers=self.get_auth_header(), json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"预测注意力失败: {response.json().get('error')}")
                return None
        except Exception as e:
            print(f"预测注意力时出错: {str(e)}")
            return None
    
    async def connect_websocket(self) -> bool:
        """连接到WebSocket服务器"""
        if not self.token:
            print("未登录，请先调用login方法")
            return False
        
        try:
            # 连接到WebSocket服务器
            self.websocket = await websockets.connect(self.ws_url)
            
            # 发送认证消息
            auth_message = {
                "type": "auth",
                "token": self.token
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # 等待认证结果
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "auth_success":
                self.client_id = response_data.get("client_id")
                print(f"WebSocket连接成功! 客户端ID: {self.client_id}")
                return True
            else:
                print(f"WebSocket认证失败: {response_data.get('message')}")
                return False
        
        except Exception as e:
            print(f"WebSocket连接时出错: {str(e)}")
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            return False
    
    async def start_simulation(self, seconds: int = 3600, sample_rate: int = 256,
                             update_interval: float = 1.0, 
                             attention_level: Optional[float] = None) -> bool:
        """启动模拟EEG数据流"""
        if not self.websocket:
            print("WebSocket未连接，请先调用connect_websocket方法")
            return False
        
        try:
            # 发送启动模拟消息
            message = {
                "type": "start_simulation",
                "seconds": seconds,
                "sample_rate": sample_rate,
                "update_interval": update_interval,
                "attention_level": attention_level
            }
            await self.websocket.send(json.dumps(message))
            
            # 等待确认消息
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "simulation_started":
                print(f"模拟数据流已启动: {response_data.get('params')}")
                return True
            else:
                print(f"启动模拟数据流失败: {response_data.get('message')}")
                return False
                
        except Exception as e:
            print(f"启动模拟数据流时出错: {str(e)}")
            return False
    
    async def subscribe_subject(self, subject_id: str, update_interval: float = 1.0,
                             preprocess: bool = True) -> bool:
        """订阅特定被试的数据"""
        if not self.websocket:
            print("WebSocket未连接，请先调用connect_websocket方法")
            return False
        
        try:
            # 发送订阅消息
            message = {
                "type": "subscribe_subject",
                "subject_id": subject_id,
                "update_interval": update_interval,
                "preprocess": preprocess
            }
            await self.websocket.send(json.dumps(message))
            
            # 等待确认消息
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "subject_subscribed":
                print(f"已订阅被试 {subject_id} 的数据: {response_data.get('params')}")
                return True
            else:
                print(f"订阅被试数据失败: {response_data.get('message')}")
                return False
                
        except Exception as e:
            print(f"订阅被试数据时出错: {str(e)}")
            return False
    
    async def stop_simulation(self) -> bool:
        """停止模拟EEG数据流"""
        if not self.websocket:
            print("WebSocket未连接，请先调用connect_websocket方法")
            return False
        
        try:
            # 发送停止模拟消息
            message = {
                "type": "stop_simulation"
            }
            await self.websocket.send(json.dumps(message))
            
            # 等待确认消息
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "simulation_stopped":
                print("模拟数据流已停止")
                return True
            else:
                print(f"停止模拟数据流失败: {response_data.get('message')}")
                return False
                
        except Exception as e:
            print(f"停止模拟数据流时出错: {str(e)}")
            return False
    
    async def receive_data(self, max_samples: int = 60, plot_result: bool = True) -> List[Dict[str, Any]]:
        """接收数据并可选择性地绘制结果"""
        if not self.websocket:
            print("WebSocket未连接，请先调用connect_websocket方法")
            return []
        
        received_data = []
        self.last_attention_scores = []
        
        try:
            # 初始化绘图
            if plot_result:
                plt.figure(figsize=(14, 8))
                plt.ion()  # 开启交互模式
                
                # 创建两个子图
                ax1 = plt.subplot(2, 1, 1)  # EEG数据
                ax2 = plt.subplot(2, 1, 2)  # 注意力分数
            
            i = 0
            while i < max_samples:
                # 接收数据
                response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                
                # 处理不同类型的消息
                if response_data.get("type") == "eeg_data" or response_data.get("type") == "subject_data":
                    received_data.append(response_data)
                    
                    # 提取注意力分数
                    attention = response_data.get("attention", {})
                    score = attention.get("score", 0)
                    level = attention.get("level", "unknown")
                    
                    print(f"接收数据 #{i+1} - 注意力分数: {score:.2f} ({level})")
                    self.last_attention_scores.append(score)
                    
                    # 绘制结果
                    if plot_result and i % 2 == 0:  # 每2个样本更新一次图像
                        self._update_plot(response_data, ax1, ax2)
                    
                    i += 1
                    
                elif response_data.get("type") == "error":
                    print(f"接收数据时出错: {response_data.get('message')}")
                    break
                    
                elif response_data.get("type") == "subject_data_complete":
                    print(f"被试数据传输完成: {response_data.get('subject_id')}")
                    break
                
                # 简短延迟
                await asyncio.sleep(0.01)
            
            # 关闭绘图
            if plot_result:
                plt.ioff()
                plt.show()
            
            return received_data
            
        except asyncio.TimeoutError:
            print("接收数据超时")
            return received_data
        except Exception as e:
            print(f"接收数据时出错: {str(e)}")
            return received_data
    
    def _update_plot(self, data: Dict[str, Any], ax1, ax2):
        """更新绘图"""
        # 清除旧数据
        ax1.clear()
        ax2.clear()
        
        # 绘制EEG数据
        eeg_data = np.array(data.get("data", []))
        times = np.array(data.get("times", []))
        channel_names = data.get("channel_names", [])
        
        if len(eeg_data) > 0 and len(times) > 0:
            # 只绘制前6个通道
            channels_to_plot = min(6, len(eeg_data))
            for i in range(channels_to_plot):
                ax1.plot(times, eeg_data[i] + i*50, label=channel_names[i] if i < len(channel_names) else f"Ch{i}")
            
            ax1.set_title("EEG数据")
            ax1.set_xlabel("时间 (秒)")
            ax1.set_ylabel("幅度 (µV)")
            ax1.legend(loc="upper right")
            ax1.grid(True)
        
        # 绘制注意力分数历史
        if len(self.last_attention_scores) > 0:
            x = np.arange(len(self.last_attention_scores))
            ax2.plot(x, self.last_attention_scores, 'r-', linewidth=2)
            ax2.fill_between(x, 0, self.last_attention_scores, color='r', alpha=0.3)
            
            # 绘制注意力级别区域
            ax2.axhspan(0, 20, alpha=0.2, color='blue', label='非常低')
            ax2.axhspan(20, 40, alpha=0.2, color='cyan', label='低')
            ax2.axhspan(40, 60, alpha=0.2, color='green', label='中等')
            ax2.axhspan(60, 80, alpha=0.2, color='yellow', label='高')
            ax2.axhspan(80, 100, alpha=0.2, color='red', label='非常高')
            
            ax2.set_title("注意力分数历史")
            ax2.set_xlabel("样本")
            ax2.set_ylabel("注意力分数")
            ax2.set_ylim(0, 100)
            ax2.legend(loc="upper right")
            ax2.grid(True)
        
        # 刷新图形
        plt.tight_layout()
        plt.pause(0.01)
    
    async def close(self):
        """关闭WebSocket连接"""
        if self.websocket:
            # 如果有活跃的模拟，先停止它
            if self.client_id:
                await self.stop_simulation()
            
            # 关闭连接
            await self.websocket.close()
            self.websocket = None
            print("WebSocket连接已关闭")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="脑注意力系统API客户端示例")
    parser.add_argument("--rest-url", default=REST_API_URL, help="REST API服务器URL")
    parser.add_argument("--ws-url", default=WEBSOCKET_API_URL, help="WebSocket服务器URL")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="用户名")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="密码")
    parser.add_argument("--mode", choices=["rest", "websocket"], default="websocket", 
                      help="API模式: rest或websocket")
    parser.add_argument("--subject", help="要处理的被试ID")
    parser.add_argument("--attention", type=float, help="模拟的注意力水平 (0-100)")
    args = parser.parse_args()
    
    # 创建客户端
    client = BrainAttentionAPIClient(args.rest_url, args.ws_url)
    
    # 登录
    if not client.login(args.username, args.password):
        return
    
    if args.mode == "rest":
        # REST API示例
        demo_rest_api(client, args.subject, args.attention)
    else:
        # WebSocket API示例
        await demo_websocket_api(client, args.subject, args.attention)

def demo_rest_api(client, subject_id=None, attention_level=None):
    """演示REST API功能"""
    print("\n=== REST API演示 ===\n")
    
    # 获取所有被试
    subjects = client.get_subjects()
    
    if not subject_id and subjects:
        # 如果未指定被试，使用第一个被试
        subject_id = subjects[0]
    
    if subject_id:
        print(f"\n获取被试 {subject_id} 的数据...")
        subject_data = client.get_subject_data(subject_id)
        if subject_data:
            info = subject_data.get("info", {})
            print(f"被试信息:")
            print(f"  - 采样率: {info.get('sample_rate', 0)} Hz")
            print(f"  - 通道数: {info.get('channel_count', 0)}")
            print(f"  - 数据时长: {info.get('duration', 0):.2f} 秒")
            print(f"  - 注意力分数: {subject_data.get('attention_score', 0):.2f}")
        
        print(f"\n获取被试 {subject_id} 的原始EEG数据片段...")
        raw_data = client.get_subject_raw_data(subject_id, start=0, duration=5)
        if raw_data:
            print(f"  - 采样率: {raw_data.get('sample_rate', 0)} Hz")
            print(f"  - 通道数: {len(raw_data.get('channel_names', []))}")
            print(f"  - 数据点数: {len(raw_data.get('times', []))}")
            
            # 简单绘制数据
            plt.figure(figsize=(10, 6))
            times = raw_data.get("times", [])
            data = raw_data.get("data", [])
            channel_names = raw_data.get("channel_names", [])
            
            for i, ch_data in enumerate(data[:6]):  # 只绘制前6个通道
                plt.plot(times, ch_data + i*50, label=channel_names[i] if i < len(channel_names) else f"Ch{i}")
            
            plt.title(f"被试 {subject_id} 的EEG数据")
            plt.xlabel("时间 (秒)")
            plt.ylabel("幅度 (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()
    
    print("\n生成模拟的EEG数据...")
    simulated_data = client.simulate_eeg_data(seconds=5, attention_level=attention_level)
    if simulated_data:
        print(f"  - 采样率: {simulated_data.get('sample_rate', 0)} Hz")
        print(f"  - 通道数: {len(simulated_data.get('channel_names', []))}")
        print(f"  - 注意力分数: {simulated_data.get('attention_score', 0):.2f}")
        
        # 使用模拟数据预测注意力
        print("\n使用模拟数据预测注意力...")
        prediction = client.predict_attention(
            simulated_data.get("data", []), 
            simulated_data.get("channel_names", []), 
            simulated_data.get("sample_rate", 0)
        )
        
        if prediction:
            print(f"  - 注意力分数: {prediction.get('attention_score', 0):.2f}")
            print(f"  - 注意力级别: {prediction.get('attention_level', 'unknown')}")

async def demo_websocket_api(client, subject_id=None, attention_level=None):
    """演示WebSocket API功能"""
    print("\n=== WebSocket API演示 ===\n")
    
    try:
        # 连接到WebSocket服务器
        if not await client.connect_websocket():
            return
        
        # 获取所有被试
        if not subject_id:
            subjects = client.get_subjects()
            if subjects:
                subject_id = subjects[0]
        
        if subject_id:
            # 订阅被试数据
            print(f"\n订阅被试 {subject_id} 的数据流...")
            if await client.subscribe_subject(subject_id, update_interval=0.5):
                # 接收一段时间的数据
                print("\n接收被试数据...")
                received_data = await client.receive_data(max_samples=30, plot_result=True)
                print(f"接收到 {len(received_data)} 个数据包")
        else:
            # 启动模拟数据流
            print("\n启动模拟数据流...")
            if await client.start_simulation(seconds=60, update_interval=0.5, attention_level=attention_level):
                # 接收一段时间的数据
                print("\n接收模拟数据...")
                received_data = await client.receive_data(max_samples=30, plot_result=True)
                print(f"接收到 {len(received_data)} 个数据包")
            
            # 停止模拟
            print("\n停止模拟数据流...")
            await client.stop_simulation()
    
    finally:
        # 关闭连接
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 