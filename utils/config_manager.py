#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
"""

import os
import json
import logging

class ConfigManager:
    """配置管理类，用于加载和管理系统配置"""
    
    def __init__(self, config_path="config/default.json"):
        """
        初始化配置管理器
        
        参数:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger('attention_system')
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """
        从文件加载配置
        
        返回:
            配置字典
        """
        # 默认配置
        default_config = {
            "app_name": "儿童注意力康复训练系统",
            "version": "1.0.0",
            "device": {
                "type": "OpenBCI",
                "channels": 8,
                "sample_rate": 256,
                "com_port": "COM3",  # Windows 默认，需要根据实际情况调整
                "mac_port": "/dev/cu.usbserial-DM00Q0QQ",  # Mac 默认，需要根据实际情况调整
                "timeout": 5,
                "channel_names": ["TP9", "TP10", "AF7", "AF8", "T7", "T8", "O1", "O2"]
            },
            "signal_processing": {
                "wavelet": "db4",
                "decomposition_level": 5,
                "frequency_bands": {
                    "delta": [0.5, 4],
                    "theta": [4, 8],
                    "alpha": [8, 13],
                    "beta": [13, 30],
                    "gamma": [30, 50]
                },
                "window_size": 4,  # 秒
                "overlap": 0.5,    # 重叠比例
                "baseline_duration": 60  # 基线校准时间（秒）
            },
            "attention_model": {
                "model_type": "random_forest",
                "model_path": "models/attention_model.pkl",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "attention_levels": ["very_low", "low", "medium", "high", "very_high"],
                "update_interval": 1  # 注意力水平更新间隔（秒）
            },
            "games": {
                "space_baby": {
                    "difficulty_levels": 5,
                    "min_duration": 180,  # 最短游戏时间（秒）
                    "speed_factor": 1.5,  # 速度增长因子
                    "obstacle_count": 10  # 障碍物数量
                },
                "magic_forest": {
                    "difficulty_levels": 5,
                    "min_duration": 240,
                    "distraction_factor": 2.0,  # 干扰选项增加因子
                    "task_count": 15  # 任务总数
                },
                "color_puzzle": {
                    "difficulty_levels": 5,
                    "min_duration": 300,
                    "rotation_speed_factor": 1.2,  # 旋转速度因子
                    "puzzle_complexity": 3  # 拼图复杂度级别
                }
            },
            "training": {
                "session_duration": 30,  # 默认训练时长（分钟）
                "break_interval": 10,    # 休息提醒间隔（分钟）
                "daily_target": 2,       # 每日目标训练次数
                "adaptation_threshold": 0.75  # 难度自适应阈值
            },
            "ui": {
                "theme": "default",
                "fullscreen": False,
                "language": "zh_CN",
                "animation_speed": 1.0,
                "volume": 0.7
            },
            "data": {
                "save_raw_eeg": True,
                "save_processed_features": True,
                "save_attention_scores": True,
                "data_dir": "data",
                "backup_interval": 7  # 数据备份间隔（天）
            }
        }
        
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # 检查配置文件是否存在
            if not os.path.exists(self.config_path):
                # 创建默认配置文件
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=4)
                self.logger.info(f"已创建默认配置文件: {self.config_path}")
                return default_config
            
            # 从文件加载配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.logger.info(f"已加载配置文件: {self.config_path}")
                
                # 检查配置是否需要更新（与默认配置合并）
                updated = False
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                        updated = True
                
                if updated:
                    self._save_config(config)
                    self.logger.info("已使用默认值更新配置文件")
                
                return config
                
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return default_config
    
    def _save_config(self, config):
        """
        保存配置到文件
        
        参数:
            config: 配置字典
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.logger.info(f"已保存配置到: {self.config_path}")
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
    
    def get_config(self):
        """
        获取当前配置
        
        返回:
            配置字典
        """
        return self.config
    
    def update_config(self, new_config):
        """
        更新配置并保存
        
        参数:
            new_config: 新的配置字典
            
        返回:
            更新后的配置字典
        """
        self.config.update(new_config)
        self._save_config(self.config)
        return self.config
    
    def get_value(self, key_path, default=None):
        """
        获取指定路径的配置值
        
        参数:
            key_path: 配置键路径，例如 "device.channels"
            default: 如果路径不存在，返回的默认值
            
        返回:
            配置值或默认值
        """
        try:
            value = self.config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default 