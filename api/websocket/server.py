#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebSocket服务器
提供实时EEG数据和注意力水平的传输功能
"""

import os
import sys
import json
import logging
import asyncio
import datetime
import jwt
import uuid
from websockets import serve, WebSocketServerProtocol
from typing import Dict, Set, Any, Optional

# 确保可以导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.hbn_eeg_loader import HBNEEGLoader

# 配置
SECRET_KEY = os.environ.get('WS_SECRET_KEY', 'brain-attention-system-secret')
DATA_DIR = os.environ.get('DATA_DIR', 'data/eeg/hbn')
PROCESSED_DIR = os.environ.get('PROCESSED_DIR', 'data/eeg/processed')
TOKEN_EXPIRATION = int(os.environ.get('TOKEN_EXPIRATION', 24 * 60 * 60))  # 默认24小时

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化EEG数据加载器
eeg_loader = HBNEEGLoader(
    data_dir=DATA_DIR,
    processed_dir=PROCESSED_DIR,
    logger=logger
)

# 客户端连接管理
active_connections: Dict[str, WebSocketServerProtocol] = {}  # 存储活跃的WebSocket连接
authorized_clients: Set[str] = set()  # 存储已认证的客户端ID

# 模拟数据会话管理
simulation_sessions: Dict[str, Dict[str, Any]] = {}  # 存储模拟数据会话

# 验证令牌
async def validate_token(token: str) -> Optional[str]:
    """验证令牌并返回用户ID"""
    try:
        # 解码token
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        
        # 检查token是否过期
        if datetime.datetime.fromtimestamp(payload['exp']) < datetime.datetime.now():
            logger.warning("令牌已过期")
            return None
            
        # 返回用户ID
        return payload['user_id']
        
    except jwt.InvalidTokenError as e:
        logger.warning(f"无效的令牌: {str(e)}")
        return None

# 处理客户端连接
async def handle_client(websocket: WebSocketServerProtocol, path: str):
    """处理新的WebSocket连接"""
    # 生成客户端ID
    client_id = str(uuid.uuid4())
    active_connections[client_id] = websocket
    logger.info(f"新客户端连接: {client_id}")
    
    try:
        # 等待认证消息
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                # 处理认证
                if message_type == 'auth':
                    token = data.get('token')
                    if not token:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': '缺少认证令牌'
                        }))
                        continue
                    
                    user_id = await validate_token(token)
                    if not user_id:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': '无效的认证令牌'
                        }))
                        continue
                    
                    # 认证成功
                    authorized_clients.add(client_id)
                    await websocket.send(json.dumps({
                        'type': 'auth_success',
                        'client_id': client_id,
                        'user_id': user_id
                    }))
                    logger.info(f"客户端 {client_id} 认证成功: 用户 {user_id}")
                    continue
                
                # 检查客户端是否已认证
                if client_id not in authorized_clients:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': '未认证，请先进行认证'
                    }))
                    continue
                
                # 处理其他消息类型
                if message_type == 'start_simulation':
                    # 启动模拟EEG数据流
                    await handle_start_simulation(websocket, client_id, data)
                    
                elif message_type == 'stop_simulation':
                    # 停止模拟EEG数据流
                    await handle_stop_simulation(client_id)
                    
                elif message_type == 'subscribe_subject':
                    # 订阅特定被试的数据
                    await handle_subscribe_subject(websocket, client_id, data)
                    
                elif message_type == 'unsubscribe_subject':
                    # 取消订阅特定被试的数据
                    subject_id = data.get('subject_id')
                    if subject_id and client_id in simulation_sessions:
                        if simulation_sessions[client_id].get('subject_id') == subject_id:
                            await handle_stop_simulation(client_id)
                
                else:
                    # 未知消息类型
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'未知的消息类型: {message_type}'
                    }))
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': '无效的JSON格式'
                }))
    
    except Exception as e:
        logger.error(f"处理客户端消息时出错: {str(e)}")
    finally:
        # 清理资源
        if client_id in active_connections:
            del active_connections[client_id]
        if client_id in authorized_clients:
            authorized_clients.remove(client_id)
        if client_id in simulation_sessions:
            await handle_stop_simulation(client_id)
        logger.info(f"客户端断开连接: {client_id}")

# 处理开始模拟请求
async def handle_start_simulation(websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
    """处理开始模拟EEG数据的请求"""
    # 如果已经有模拟会话，先停止它
    if client_id in simulation_sessions:
        await handle_stop_simulation(client_id)
    
    # 获取参数
    seconds = data.get('seconds', 3600)  # 默认模拟1小时
    sample_rate = data.get('sample_rate', 256)
    update_interval = data.get('update_interval', 1.0)  # 默认每秒更新一次
    attention_level = data.get('attention_level')  # 可以是None（随机）或0-100之间的值
    
    # 创建模拟会话
    simulation_task = asyncio.create_task(
        simulate_eeg_stream(websocket, sample_rate, attention_level, update_interval)
    )
    
    simulation_sessions[client_id] = {
        'task': simulation_task,
        'start_time': datetime.datetime.now(),
        'params': {
            'seconds': seconds,
            'sample_rate': sample_rate,
            'update_interval': update_interval,
            'attention_level': attention_level
        }
    }
    
    # 发送确认消息
    await websocket.send(json.dumps({
        'type': 'simulation_started',
        'client_id': client_id,
        'params': {
            'seconds': seconds,
            'sample_rate': sample_rate,
            'update_interval': update_interval,
            'attention_level': attention_level
        }
    }))
    
    logger.info(f"客户端 {client_id} 开始模拟EEG数据流")

# 处理停止模拟请求
async def handle_stop_simulation(client_id: str):
    """处理停止模拟EEG数据的请求"""
    if client_id in simulation_sessions:
        # 取消任务
        simulation_sessions[client_id]['task'].cancel()
        
        try:
            # 等待任务完成
            await simulation_sessions[client_id]['task']
        except asyncio.CancelledError:
            # 任务已取消
            pass
        
        # 移除会话
        del simulation_sessions[client_id]
        
        # 如果客户端仍然连接，发送确认消息
        if client_id in active_connections:
            await active_connections[client_id].send(json.dumps({
                'type': 'simulation_stopped',
                'client_id': client_id
            }))
        
        logger.info(f"客户端 {client_id} 停止模拟EEG数据流")

# 处理订阅被试请求
async def handle_subscribe_subject(websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
    """处理订阅特定被试数据的请求"""
    subject_id = data.get('subject_id')
    if not subject_id:
        await websocket.send(json.dumps({
            'type': 'error',
            'message': '缺少被试ID'
        }))
        return
    
    update_interval = data.get('update_interval', 1.0)  # 默认每秒更新一次
    preprocess = data.get('preprocess', True)
    
    # 加载被试数据
    data_dict = eeg_loader.load_subject_data(subject_id, preprocess=preprocess)
    
    if data_dict is None:
        await websocket.send(json.dumps({
            'type': 'error',
            'message': f"找不到被试 {subject_id} 的数据"
        }))
        return
    
    # 如果已经有模拟会话，先停止它
    if client_id in simulation_sessions:
        await handle_stop_simulation(client_id)
    
    # 从MNE Raw对象中提取必要信息
    raw = data_dict['raw']
    sfreq = raw.info['sfreq']
    
    # 创建被试数据流任务
    simulation_task = asyncio.create_task(
        stream_subject_data(websocket, subject_id, raw, data_dict['features'], update_interval)
    )
    
    simulation_sessions[client_id] = {
        'task': simulation_task,
        'start_time': datetime.datetime.now(),
        'subject_id': subject_id,
        'params': {
            'update_interval': update_interval,
            'preprocess': preprocess
        }
    }
    
    # 发送确认消息
    await websocket.send(json.dumps({
        'type': 'subject_subscribed',
        'client_id': client_id,
        'subject_id': subject_id,
        'params': {
            'update_interval': update_interval,
            'sample_rate': sfreq,
            'channel_count': len(raw.ch_names),
            'channels': raw.ch_names
        }
    }))
    
    logger.info(f"客户端 {client_id} 订阅被试 {subject_id} 的数据流")

# 模拟EEG数据流
async def simulate_eeg_stream(websocket: WebSocketServerProtocol, 
                             sample_rate: int, 
                             attention_level: Optional[float], 
                             update_interval: float):
    """生成和发送模拟的EEG数据流"""
    segment_duration = update_interval  # 每次发送的数据段时长（秒）
    
    try:
        while True:
            # 生成一段模拟数据
            raw, features = eeg_loader.generate_simulated_data(
                seconds=segment_duration,
                sfreq=sample_rate,
                attention_level=attention_level
            )
            
            # 提取数据
            data, times = raw[:, :]
            
            # 计算注意力分数
            attention_score = eeg_loader.get_attention_score(features)
            
            # 确定注意力级别
            if attention_score >= 80:
                level = "very_high"
            elif attention_score >= 60:
                level = "high"
            elif attention_score >= 40:
                level = "medium"
            elif attention_score >= 20:
                level = "low"
            else:
                level = "very_low"
            
            # 构建消息
            message = {
                'type': 'eeg_data',
                'timestamp': datetime.datetime.now().isoformat(),
                'sample_rate': sample_rate,
                'channel_names': raw.ch_names,
                'times': times.tolist(),
                'data': data.tolist(),
                'features': features,
                'attention': {
                    'score': attention_score,
                    'level': level
                },
                'segment_duration': segment_duration
            }
            
            # 发送消息
            await websocket.send(json.dumps(message))
            
            # 等待下一个更新间隔
            await asyncio.sleep(update_interval)
            
    except asyncio.CancelledError:
        # 任务被取消
        raise
    except Exception as e:
        logger.error(f"模拟EEG数据流时出错: {str(e)}")
        if websocket.open:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f"模拟数据流出错: {str(e)}"
            }))

# 流式传输被试数据
async def stream_subject_data(websocket: WebSocketServerProtocol, 
                             subject_id: str,
                             raw,
                             features: Dict[str, Any],
                             update_interval: float):
    """流式传输被试EEG数据"""
    # 从MNE Raw对象中提取必要信息
    sfreq = raw.info['sfreq']
    total_duration = raw.times[-1]  # 数据总时长（秒）
    
    # 流式传输参数
    segment_duration = update_interval  # 每次发送的数据段时长（秒）
    current_position = 0.0  # 当前位置（秒）
    
    try:
        while current_position < total_duration:
            # 计算当前段的起止位置
            start_sample = int(current_position * sfreq)
            end_sample = int((current_position + segment_duration) * sfreq)
            
            # 防止超出范围
            if end_sample >= len(raw.times):
                end_sample = len(raw.times) - 1
            
            # 提取当前段数据
            data, times = raw[:, start_sample:end_sample]
            
            # 如果到达文件末尾，设置适当的标志
            is_end_of_file = end_sample >= len(raw.times) - 1
            
            # 计算注意力分数
            # 实际应用中，可能需要基于当前段数据重新计算特征
            attention_score = eeg_loader.get_attention_score(features)
            
            # 确定注意力级别
            if attention_score >= 80:
                level = "very_high"
            elif attention_score >= 60:
                level = "high"
            elif attention_score >= 40:
                level = "medium"
            elif attention_score >= 20:
                level = "low"
            else:
                level = "very_low"
            
            # 构建消息
            message = {
                'type': 'subject_data',
                'subject_id': subject_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'current_position': current_position,
                'total_duration': total_duration,
                'is_end_of_file': is_end_of_file,
                'sample_rate': sfreq,
                'channel_names': raw.ch_names,
                'times': times.tolist(),
                'data': data.tolist(),
                'features': features,
                'attention': {
                    'score': attention_score,
                    'level': level
                }
            }
            
            # 发送消息
            await websocket.send(json.dumps(message))
            
            # 更新位置
            current_position += segment_duration
            
            # 如果到达文件末尾，则结束
            if is_end_of_file:
                # 发送完成消息
                await websocket.send(json.dumps({
                    'type': 'subject_data_complete',
                    'subject_id': subject_id,
                    'timestamp': datetime.datetime.now().isoformat()
                }))
                break
            
            # 等待下一个更新间隔
            await asyncio.sleep(update_interval)
            
    except asyncio.CancelledError:
        # 任务被取消
        raise
    except Exception as e:
        logger.error(f"流式传输被试 {subject_id} 数据时出错: {str(e)}")
        if websocket.open:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f"数据流出错: {str(e)}"
            }))

# 启动服务器
async def main():
    # 设置主机和端口
    host = os.environ.get('WS_HOST', '0.0.0.0')
    port = int(os.environ.get('WS_PORT', 5001))
    
    # 扫描可用的EEG文件
    logger.info("扫描可用的HBN EEG文件...")
    subjects = eeg_loader.get_subject_list()
    logger.info(f"找到 {len(subjects)} 个被试。")
    
    # 启动WebSocket服务器
    server = await serve(handle_client, host, port)
    logger.info(f"WebSocket服务器正在运行，监听 {host}:{port}")
    
    # 保持服务器运行
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main()) 