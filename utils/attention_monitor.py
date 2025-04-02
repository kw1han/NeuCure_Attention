#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
注意力监测器模块
整合EEG采集、处理和分类，实时监测用户的注意力水平
"""

import time
import numpy as np
import threading
import logging
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime
import os
import json
import pandas as pd

from utils.eeg_acquisition import EEGAcquisition
from utils.eeg_processor import EEGProcessor
from models.attention_classifier import AttentionClassifier

class AttentionMonitor(QObject):
    """注意力监测器类，整合采集、处理和分类"""
    
    # 定义信号
    attention_updated = pyqtSignal(dict)  # 注意力水平更新信号
    raw_eeg_updated = pyqtSignal(np.ndarray)  # 原始EEG更新信号
    connection_status_changed = pyqtSignal(bool, str)  # 连接状态变化信号
    calibration_progress = pyqtSignal(int)  # 校准进度信号
    calibration_complete = pyqtSignal(bool, str)  # 校准完成信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, config):
        """
        初始化注意力监测器
        
        参数:
            config: 配置信息字典
        """
        super().__init__()
        
        self.logger = logging.getLogger('attention_system')
        self.config = config
        
        # 更新间隔
        self.update_interval = config['attention_model']['update_interval']
        
        # 初始化组件
        self.eeg_acquisition = EEGAcquisition(config, self._handle_eeg_sample)
        self.eeg_processor = EEGProcessor(config)
        self.attention_classifier = AttentionClassifier(config)
        
        # 线程控制
        self.running = False
        self.monitor_thread = None
        self.processing_lock = threading.Lock()
        
        # 数据存储
        self.data_dir = config['data']['data_dir']
        self.save_raw_eeg = config['data']['save_raw_eeg']
        self.save_features = config['data']['save_processed_features']
        self.save_attention = config['data']['save_attention_scores']
        
        # 存储缓冲区
        self.raw_buffer = []
        self.feature_buffer = []
        self.attention_buffer = []
        self.max_buffer_size = 60 * 30  # 30分钟的数据
        
        # 会话ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 状态追踪
        self.last_update_time = 0
        self.last_attention_level = None
        self.attention_history = []
        self.baseline_attention = None
        
        # 校准状态
        self.is_calibrating = False
        self.calibration_data = []
        self.baseline_duration = config['signal_processing']['baseline_duration']
        
        self.logger.info("注意力监测器初始化完成")
    
    def start(self):
        """
        启动注意力监测
        
        返回:
            是否成功启动
        """
        if self.running:
            self.logger.warning("注意力监测已在运行中")
            return False
            
        self.logger.info("启动注意力监测")
        
        # 创建数据存储目录
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 连接到设备
        if not self.eeg_acquisition.connect():
            error_msg = "无法连接到EEG设备"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
        
        # 启动数据采集
        if not self.eeg_acquisition.start():
            error_msg = "无法启动EEG数据采集"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
        
        # 设置状态
        self.running = True
        self.last_update_time = time.time()
        
        # 启动监测线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # 发送连接状态信号
        device_info = self.eeg_acquisition.get_device_info()
        status_msg = f"已连接到{device_info['device_type']}设备，{device_info['channels']}通道，{device_info['sample_rate']}Hz"
        self.connection_status_changed.emit(True, status_msg)
        
        self.logger.info("注意力监测已启动")
        return True
    
    def stop(self):
        """
        停止注意力监测
        
        返回:
            是否成功停止
        """
        if not self.running:
            self.logger.warning("注意力监测未在运行中")
            return False
            
        self.logger.info("停止注意力监测")
        
        # 设置状态
        self.running = False
        
        # 停止数据采集
        self.eeg_acquisition.stop()
        
        # 等待监测线程结束
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # 保存剩余数据
        self._save_session_data()
        
        # 发送连接状态信号
        self.connection_status_changed.emit(False, "已断开连接")
        
        self.logger.info("注意力监测已停止")
        return True
    
    def start_calibration(self):
        """
        开始注意力基线校准
        
        返回:
            是否成功启动校准
        """
        if not self.running:
            error_msg = "无法开始校准：监测器未运行"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
            
        if self.is_calibrating:
            self.logger.warning("校准已在进行中")
            return False
            
        self.logger.info("开始注意力基线校准")
        
        # 设置校准状态
        self.is_calibrating = True
        self.calibration_data = []
        
        # 发送校准开始信号
        self.calibration_progress.emit(0)
        
        self.logger.info(f"校准将持续{self.baseline_duration}秒")
        return True
    
    def _handle_eeg_sample(self, sample):
        """
        处理单个EEG样本
        
        参数:
            sample: EEG样本数据
        """
        # 发送原始EEG信号
        self.raw_eeg_updated.emit(sample)
        
        # 保存原始数据
        if self.save_raw_eeg:
            self.raw_buffer.append({
                'timestamp': time.time(),
                'data': sample.tolist()
            })
            
            # 限制缓冲区大小
            if len(self.raw_buffer) > self.max_buffer_size:
                self._save_raw_data()
    
    def _monitor_loop(self):
        """注意力监测主循环"""
        self.logger.info("注意力监测循环已启动")
        
        batch_size = int(self.update_interval * self.eeg_acquisition.sample_rate)
        next_update_time = time.time() + self.update_interval
        
        while self.running:
            try:
                current_time = time.time()
                
                # 检查是否需要更新注意力水平
                if current_time >= next_update_time:
                    with self.processing_lock:
                        # 获取EEG缓冲数据
                        eeg_buffer = self.eeg_acquisition.get_buffer()
                        
                        # 处理EEG数据
                        features_list = self.eeg_processor.process_eeg_batch(eeg_buffer)
                        
                        if features_list:
                            # 保存特征数据
                            if self.save_features:
                                for features in features_list:
                                    self.feature_buffer.append({
                                        'timestamp': current_time,
                                        'features': features
                                    })
                            
                            # 预测注意力水平
                            latest_features = features_list[-1]
                            attention_result = self.attention_classifier.predict(latest_features)
                            
                            # 获取平滑注意力水平
                            smoothed_attention = self.attention_classifier.get_smoothed_attention()
                            if smoothed_attention:
                                attention_result = smoothed_attention
                            
                            # 计算注意力分数
                            attention_score = self.attention_classifier.attention_score_to_numeric(attention_result)
                            
                            # 如果在校准中，收集数据
                            if self.is_calibrating:
                                self._update_calibration(attention_score)
                            else:
                                # 更新并发送注意力结果
                                self._update_attention(attention_result, attention_score)
                            
                    # 设置下一次更新时间
                    next_update_time = current_time + self.update_interval
                
                # 控制循环速率
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"注意力监测循环出错: {e}")
                error_msg = f"注意力监测错误: {str(e)}"
                self.error_occurred.emit(error_msg)
                time.sleep(1.0)  # 出错后暂停一秒再继续
        
        self.logger.info("注意力监测循环已停止")
    
    def _update_attention(self, attention_result, attention_score):
        """
        更新注意力水平结果
        
        参数:
            attention_result: 注意力分类结果
            attention_score: 注意力分数(0-100)
        """
        current_time = time.time()
        
        # 创建注意力状态对象
        attention_state = {
            'timestamp': current_time,
            'level': attention_result['level'],
            'level_idx': attention_result['level_idx'],
            'probability': attention_result['probability'],
            'score': attention_score,
            'normalized_score': self._normalize_score(attention_score),
            'raw_data': attention_result
        }
        
        # 保存到历史记录
        self.attention_history.append(attention_state)
        if len(self.attention_history) > 600:  # 保存最近10分钟的数据(按1秒一个样本计算)
            self.attention_history.pop(0)
        
        # 保存注意力数据
        if self.save_attention:
            self.attention_buffer.append(attention_state)
            
            # 限制缓冲区大小
            if len(self.attention_buffer) > self.max_buffer_size:
                self._save_attention_data()
        
        # 发送注意力更新信号
        self.attention_updated.emit(attention_state)
        
        # 更新最后状态
        self.last_attention_level = attention_result['level']
        self.last_update_time = current_time
        
        self.logger.debug(f"注意力水平更新: {attention_result['level']}, 分数: {attention_score}")
    
    def _update_calibration(self, attention_score):
        """
        更新校准进度
        
        参数:
            attention_score: 当前注意力分数
        """
        # 添加校准数据
        self.calibration_data.append(attention_score)
        
        # 计算校准进度
        elapsed_time = len(self.calibration_data) * self.update_interval
        progress = int((elapsed_time / self.baseline_duration) * 100)
        progress = min(100, progress)
        
        # 发送进度信号
        self.calibration_progress.emit(progress)
        
        # 检查校准是否完成
        if elapsed_time >= self.baseline_duration:
            self._complete_calibration()
    
    def _complete_calibration(self):
        """完成校准过程"""
        self.logger.info("完成注意力基线校准")
        
        # 计算基线注意力
        if self.calibration_data:
            self.baseline_attention = {
                'mean': np.mean(self.calibration_data),
                'median': np.median(self.calibration_data),
                'std': np.std(self.calibration_data),
                'min': np.min(self.calibration_data),
                'max': np.max(self.calibration_data),
                'timestamp': time.time()
            }
            
            # 保存基线数据
            self._save_baseline_data()
            
            self.logger.info(f"基线注意力: 均值={self.baseline_attention['mean']:.2f}, 标准差={self.baseline_attention['std']:.2f}")
            success_msg = f"校准完成，基线注意力均值: {self.baseline_attention['mean']:.2f}"
            self.calibration_complete.emit(True, success_msg)
        else:
            error_msg = "校准失败: 未收集到数据"
            self.logger.error(error_msg)
            self.calibration_complete.emit(False, error_msg)
        
        # 重置校准状态
        self.is_calibrating = False
        self.calibration_data = []
    
    def _normalize_score(self, score):
        """
        根据基线数据归一化注意力分数
        
        参数:
            score: 原始注意力分数
            
        返回:
            归一化后的分数(0-100)
        """
        if not self.baseline_attention:
            return score
            
        # 获取基线数据
        baseline_mean = self.baseline_attention['mean']
        baseline_std = self.baseline_attention['std']
        
        # 避免除零错误
        if baseline_std == 0:
            baseline_std = 1.0
            
        # 计算z分数
        z_score = (score - baseline_mean) / baseline_std
        
        # 转换为0-100范围的分数
        normalized = 50 + (z_score * 10)  # 每个标准差对应10分变化
        
        # 限制在0-100范围内
        normalized = max(0, min(100, normalized))
        
        return round(normalized, 1)
    
    def _save_raw_data(self):
        """保存原始EEG数据"""
        if not self.raw_buffer:
            return
            
        try:
            filename = f"{self.data_dir}/raw_eeg_{self.session_id}.json"
            with open(filename, 'a') as f:
                for item in self.raw_buffer:
                    json.dump(item, f)
                    f.write('\n')
                    
            self.logger.debug(f"已保存{len(self.raw_buffer)}条原始EEG数据到{filename}")
            self.raw_buffer = []
            
        except Exception as e:
            self.logger.error(f"保存原始EEG数据失败: {e}")
    
    def _save_feature_data(self):
        """保存特征数据"""
        if not self.feature_buffer:
            return
            
        try:
            filename = f"{self.data_dir}/features_{self.session_id}.json"
            with open(filename, 'a') as f:
                for item in self.feature_buffer:
                    json.dump(item, f)
                    f.write('\n')
                    
            self.logger.debug(f"已保存{len(self.feature_buffer)}条特征数据到{filename}")
            self.feature_buffer = []
            
        except Exception as e:
            self.logger.error(f"保存特征数据失败: {e}")
    
    def _save_attention_data(self):
        """保存注意力数据"""
        if not self.attention_buffer:
            return
            
        try:
            filename = f"{self.data_dir}/attention_{self.session_id}.json"
            with open(filename, 'a') as f:
                for item in self.attention_buffer:
                    # 创建可序列化的副本
                    serializable_item = item.copy()
                    serializable_item['raw_data'] = {
                        'level': item['raw_data']['level'],
                        'level_idx': item['raw_data']['level_idx'],
                        'probability': item['raw_data']['probability']
                    }
                    json.dump(serializable_item, f)
                    f.write('\n')
                    
            self.logger.debug(f"已保存{len(self.attention_buffer)}条注意力数据到{filename}")
            self.attention_buffer = []
            
        except Exception as e:
            self.logger.error(f"保存注意力数据失败: {e}")
    
    def _save_baseline_data(self):
        """保存基线校准数据"""
        if not self.baseline_attention:
            return
            
        try:
            filename = f"{self.data_dir}/baseline_{self.session_id}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'session_id': self.session_id,
                    'timestamp': time.time(),
                    'baseline': self.baseline_attention,
                    'data': self.calibration_data
                }, f, indent=2)
                
            self.logger.info(f"已保存基线校准数据到{filename}")
            
        except Exception as e:
            self.logger.error(f"保存基线数据失败: {e}")
    
    def _save_session_data(self):
        """保存会话结束时的所有剩余数据"""
        self._save_raw_data()
        self._save_feature_data()
        self._save_attention_data()
        
        # 保存会话摘要
        try:
            filename = f"{self.data_dir}/session_summary_{self.session_id}.json"
            
            # 计算注意力统计数据
            attention_scores = [item['score'] for item in self.attention_history] if self.attention_history else []
            
            session_data = {
                'session_id': self.session_id,
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)) if hasattr(self, 'start_time') else None,
                'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                'duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'attention_stats': {
                    'mean': np.mean(attention_scores) if attention_scores else None,
                    'median': np.median(attention_scores) if attention_scores else None,
                    'std': np.std(attention_scores) if attention_scores else None,
                    'min': np.min(attention_scores) if attention_scores else None,
                    'max': np.max(attention_scores) if attention_scores else None,
                    'samples': len(attention_scores)
                },
                'device_info': self.eeg_acquisition.get_device_info()
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            self.logger.info(f"已保存会话摘要到{filename}")
            
        except Exception as e:
            self.logger.error(f"保存会话摘要失败: {e}")
    
    def get_attention_history(self):
        """
        获取注意力历史数据
        
        返回:
            注意力历史数据列表
        """
        return self.attention_history.copy()
    
    def get_current_attention(self):
        """
        获取当前注意力状态
        
        返回:
            当前注意力状态
        """
        if not self.attention_history:
            return None
        return self.attention_history[-1]
    
    def export_session_data(self, export_path):
        """
        导出会话数据到CSV文件
        
        参数:
            export_path: 导出文件路径
            
        返回:
            是否成功导出
        """
        try:
            if not self.attention_history:
                self.logger.warning("没有可导出的注意力数据")
                return False
                
            # 转换为DataFrame
            data = []
            for item in self.attention_history:
                data.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['timestamp'])),
                    'level': item['level'],
                    'score': item['score'],
                    'normalized_score': item['normalized_score'],
                    'probability': item['probability']
                })
                
            df = pd.DataFrame(data)
            
            # 保存到CSV
            df.to_csv(export_path, index=False)
            
            self.logger.info(f"已导出{len(data)}条注意力数据到{export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出会话数据失败: {e}")
            return False 