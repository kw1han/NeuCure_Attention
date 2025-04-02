#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG数据采集模块
负责连接OpenBCI设备并采集脑电数据，同时提供模拟数据生成功能用于测试
"""

import time
import numpy as np
import threading
import logging
import queue
import platform
from threading import Event

# 尝试导入OpenBCI相关库，无法导入时使用模拟模式
try:
    import pyOpenBCI
    OPENBCI_AVAILABLE = True
except ImportError:
    OPENBCI_AVAILABLE = False

class EEGAcquisition:
    """脑电数据采集类"""
    
    def __init__(self, config, callback=None):
        """
        初始化EEG采集器
        
        参数:
            config: 配置信息字典
            callback: 数据回调函数，接收新采集的数据
        """
        self.logger = logging.getLogger('attention_system')
        self.config = config
        self.callback = callback
        
        # 从配置中获取参数
        self.device_type = config['device']['type']
        self.channels = config['device']['channels']
        self.sample_rate = config['device']['sample_rate']
        self.channel_names = config['device']['channel_names']
        
        # 根据操作系统选择串口
        if platform.system() == 'Darwin':  # macOS
            self.port = config['device']['mac_port']
        else:  # Windows/Linux
            self.port = config['device']['com_port']
        
        # 连接超时时间
        self.timeout = config['device']['timeout']
        
        # 是否使用模拟数据
        self.use_simulated_data = config.get('use_simulated_data', True)
        
        # 数据缓冲队列
        self.data_queue = queue.Queue()
        
        # 线程控制
        self.running = False
        self.acquisition_thread = None
        self.processing_thread = None
        self.stop_event = Event()
        
        # 初始化数据缓冲
        self.buffer_duration = 4  # 缓冲区持续时间（秒）
        self.buffer_size = int(self.buffer_duration * self.sample_rate)
        self.data_buffer = np.zeros((self.buffer_size, self.channels))
        self.buffer_index = 0
        
        # 初始化计数器
        self.total_samples = 0
        self.start_time = None
        
        self.logger.info(f"EEG采集器初始化完成，使用{'模拟数据' if self.use_simulated_data else 'OpenBCI设备'}")
    
    def connect(self):
        """
        连接到OpenBCI设备或初始化模拟数据生成器
        
        返回:
            连接是否成功
        """
        if self.use_simulated_data or not OPENBCI_AVAILABLE:
            self.logger.info("使用模拟数据模式进行EEG采集")
            self.board = None
            return True
            
        try:
            self.logger.info(f"尝试连接到OpenBCI设备，端口: {self.port}")
            
            # 连接到OpenBCI Cyton板
            if self.device_type == 'OpenBCI':
                # 尝试连接到OpenBCI设备
                self.board = pyOpenBCI.OpenBCICyton(port=self.port, daisy=False)
                self.logger.info("已成功连接到OpenBCI Cyton设备")
                return True
            else:
                self.logger.error(f"不支持的设备类型: {self.device_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"连接OpenBCI设备失败: {e}")
            self.logger.warning("切换到模拟数据模式")
            self.use_simulated_data = True
            return True
    
    def start(self):
        """
        开始数据采集
        
        返回:
            是否成功启动
        """
        if self.running:
            self.logger.warning("采集已在运行中")
            return False
            
        self.logger.info("启动EEG数据采集")
        
        # 重置状态
        self.running = True
        self.stop_event.clear()
        self.start_time = time.time()
        self.total_samples = 0
        
        # 启动采集线程
        if self.use_simulated_data:
            self.acquisition_thread = threading.Thread(target=self._simulate_data_stream)
        else:
            self.acquisition_thread = threading.Thread(target=self._acquire_data_stream)
        
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_data_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("EEG数据采集已启动")
        return True
    
    def stop(self):
        """
        停止数据采集
        
        返回:
            是否成功停止
        """
        if not self.running:
            self.logger.warning("采集未在运行中")
            return False
            
        self.logger.info("停止EEG数据采集")
        
        # 发送停止信号
        self.running = False
        self.stop_event.set()
        
        # 等待线程结束
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=2.0)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # 如果使用的是实际设备，关闭连接
        if not self.use_simulated_data and self.board:
            try:
                self.board.stop_stream()
                self.logger.info("已关闭设备数据流")
            except Exception as e:
                self.logger.error(f"关闭设备数据流失败: {e}")
        
        # 计算统计信息
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_sample_rate = self.total_samples / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(f"EEG数据采集已停止，共采集{self.total_samples}个样本，平均采样率{avg_sample_rate:.2f}Hz")
        return True
    
    def _simulate_data_stream(self):
        """模拟脑电数据流，用于测试和演示"""
        self.logger.info("开始模拟EEG数据流")
        
        # 生成基本的Alpha, Beta, Theta和Delta节律
        alpha_freq = 10  # Hz
        beta_freq = 20   # Hz
        theta_freq = 6   # Hz
        delta_freq = 2   # Hz
        
        # 设置基本振幅
        alpha_amp = 1.0
        beta_amp = 0.5
        theta_amp = 0.7
        delta_amp = 1.5
        
        # 模拟不同通道的相位差
        phase_shifts = np.linspace(0, 2*np.pi, self.channels, endpoint=False)
        
        sample_index = 0
        
        # 模拟随时间变化的注意力水平
        attention_level = 0.5  # 起始注意力水平 (0-1)
        attention_change_rate = 0.001  # 注意力水平变化速率
        attention_direction = 1  # 变化方向 (1 增加, -1 减少)
        
        while self.running:
            try:
                # 当前时间点
                t = sample_index / self.sample_rate
                
                # 动态调整注意力水平
                attention_level += attention_direction * attention_change_rate
                if attention_level > 0.9:
                    attention_direction = -1
                elif attention_level < 0.1:
                    attention_direction = 1
                
                # 根据注意力水平调整Alpha和Beta的相对强度
                alpha_mod = alpha_amp * (1.0 - attention_level)  # 注意力高时Alpha减弱
                beta_mod = beta_amp * attention_level            # 注意力高时Beta增强
                
                # 创建一个样本
                sample = np.zeros(self.channels)
                
                for ch in range(self.channels):
                    # 各频段波形
                    alpha = alpha_mod * np.sin(2 * np.pi * alpha_freq * t + phase_shifts[ch])
                    beta = beta_mod * np.sin(2 * np.pi * beta_freq * t + phase_shifts[ch])
                    theta = theta_amp * np.sin(2 * np.pi * theta_freq * t + phase_shifts[ch])
                    delta = delta_amp * np.sin(2 * np.pi * delta_freq * t + phase_shifts[ch])
                    
                    # 添加随机噪声
                    noise = np.random.normal(0, 0.1)
                    
                    # 合成波形
                    sample[ch] = alpha + beta + theta + delta + noise
                
                # 将样本加入队列
                self.data_queue.put(sample)
                
                # 更新计数
                sample_index += 1
                self.total_samples += 1
                
                # 控制模拟采样率
                time.sleep(1.0 / self.sample_rate)
                
            except Exception as e:
                self.logger.error(f"模拟数据生成出错: {e}")
                break
        
        self.logger.info("模拟EEG数据流已停止")
    
    def _acquire_data_stream(self):
        """从OpenBCI设备获取实时数据流"""
        if not self.board:
            self.logger.error("无法启动数据流：未连接到设备")
            return
            
        self.logger.info("开始从OpenBCI设备采集数据")
        
        def handle_sample(sample):
            """处理每个OpenBCI样本"""
            if not self.running:
                return
                
            # 提取EEG数据
            eeg_data = np.array(sample.channels_data)
            
            # 将样本加入队列
            self.data_queue.put(eeg_data)
            self.total_samples += 1
        
        try:
            # 启动数据流
            self.board.start_stream(handle_sample)
            
            # 等待停止信号
            while self.running and not self.stop_event.is_set():
                time.sleep(0.1)
                
            # 停止流
            self.board.stop_stream()
            
        except Exception as e:
            self.logger.error(f"从设备采集数据时出错: {e}")
            self.running = False
            
        self.logger.info("OpenBCI数据采集已停止")
    
    def _process_data_queue(self):
        """处理数据队列中的样本"""
        self.logger.info("启动数据处理线程")
        
        while self.running:
            try:
                # 从队列中获取样本(非阻塞)
                try:
                    sample = self.data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 更新缓冲区
                self.data_buffer[self.buffer_index] = sample
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                
                # 如果有回调函数，调用它
                if self.callback is not None:
                    try:
                        self.callback(sample)
                    except Exception as e:
                        self.logger.error(f"执行回调函数时出错: {e}")
                
                # 标记任务完成
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"处理数据队列时出错: {e}")
                if not self.running:
                    break
        
        self.logger.info("数据处理线程已停止")
    
    def get_buffer(self):
        """
        获取当前数据缓冲区的副本
        
        返回:
            数据缓冲区副本，按时间排序
        """
        # 重新排列缓冲区使其按时间顺序排列
        if self.buffer_index == 0:
            # 缓冲区是空的或恰好填满一个周期
            return self.data_buffer.copy()
        else:
            # 将缓冲区分为两部分并重新排序
            return np.vstack((self.data_buffer[self.buffer_index:], self.data_buffer[:self.buffer_index]))
    
    def is_running(self):
        """
        检查采集是否正在运行
        
        返回:
            是否正在运行
        """
        return self.running
    
    def set_callback(self, callback):
        """
        设置数据回调函数
        
        参数:
            callback: 回调函数，接收单个样本作为参数
        """
        self.callback = callback
    
    def get_device_info(self):
        """
        获取设备信息
        
        返回:
            设备信息字典
        """
        info = {
            'device_type': 'Simulated' if self.use_simulated_data else self.device_type,
            'channels': self.channels,
            'sample_rate': self.sample_rate,
            'channel_names': self.channel_names,
            'running': self.running,
            'total_samples': self.total_samples
        }
        
        if self.start_time and self.running:
            info['elapsed_time'] = time.time() - self.start_time
            info['effective_sample_rate'] = self.total_samples / info['elapsed_time']
        
        return info 