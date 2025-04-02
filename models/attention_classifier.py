#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
注意力水平分类器模型
使用随机森林算法对脑电特征进行分类，评估用户的注意力水平
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

class AttentionClassifier:
    """注意力水平分类器"""
    
    def __init__(self, config):
        """
        初始化注意力分类器
        
        参数:
            config: 配置信息字典
        """
        self.logger = logging.getLogger('attention_system')
        self.config = config
        
        # 获取模型配置
        model_config = config['attention_model']
        self.model_path = model_config['model_path']
        self.model_type = model_config['model_type']
        self.n_estimators = model_config['n_estimators']
        self.max_depth = model_config['max_depth']
        self.random_state = model_config['random_state']
        self.attention_levels = model_config['attention_levels']
        
        # 预处理工具
        self.scaler = StandardScaler()
        
        # 初始化模型
        self._init_model()
        
        # 历史预测结果
        self.history = []
        self.max_history_size = 10  # 保存最近10次的预测结果用于平滑
        
        self.logger.info("注意力分类器初始化完成")
    
    def _init_model(self):
        """初始化模型"""
        # 检查是否有保存的模型
        if os.path.exists(self.model_path):
            try:
                self.model = load(self.model_path)
                self.logger.info(f"已加载现有模型: {self.model_path}")
                self.is_trained = True
            except Exception as e:
                self.logger.error(f"加载模型失败: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """创建新模型"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.logger.warning(f"未知的模型类型: {self.model_type}，使用默认随机森林")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.is_trained = False
        self.logger.info("已创建新的注意力分类模型")
    
    def train(self, features, labels):
        """
        训练注意力分类模型
        
        参数:
            features: 特征数据，Dict或DataFrame
            labels: 标签数据
            
        返回:
            训练精度
        """
        try:
            # 转换特征为DataFrame
            if isinstance(features, list):
                features_df = pd.DataFrame(features)
            else:
                features_df = features
                
            # 特征标准化
            X = self.scaler.fit_transform(features_df)
            y = np.array(labels)
            
            # 训练测试集划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            # 训练模型
            start_time = time.time()
            self.model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"模型训练完成，耗时: {train_time:.2f}秒，测试精度: {accuracy:.4f}")
            
            # 输出分类报告
            report = classification_report(y_test, y_pred, target_names=self.attention_levels)
            self.logger.debug(f"分类报告:\n{report}")
            
            # 保存模型
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            dump(self.model, self.model_path)
            self.logger.info(f"模型已保存到: {self.model_path}")
            
            self.is_trained = True
            return accuracy
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return 0.0
    
    def predict(self, features):
        """
        预测注意力水平
        
        参数:
            features: 特征数据
            
        返回:
            注意力水平类别和概率
        """
        try:
            if not self.is_trained:
                self.logger.warning("模型尚未训练，返回默认注意力水平")
                return {
                    'level': self.attention_levels[len(self.attention_levels) // 2],
                    'level_idx': len(self.attention_levels) // 2,
                    'probability': 1.0 / len(self.attention_levels),
                    'all_probs': {level: 1.0 / len(self.attention_levels) for level in self.attention_levels}
                }
            
            # 转换特征为DataFrame并标准化
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, list) and isinstance(features[0], dict):
                features = pd.DataFrame(features)
                
            X = self.scaler.transform(features)
            
            # 预测类别
            y_pred = self.model.predict(X)[0]
            
            # 预测概率
            y_proba = self.model.predict_proba(X)[0]
            
            # 获取类别索引
            level_idx = self.attention_levels.index(y_pred) if y_pred in self.attention_levels else 0
            
            # 创建结果字典
            result = {
                'level': y_pred,
                'level_idx': level_idx,
                'probability': max(y_proba),
                'all_probs': {self.attention_levels[i]: y_proba[i] for i in range(len(self.attention_levels))}
            }
            
            # 添加到历史记录
            self.history.append(result)
            if len(self.history) > self.max_history_size:
                self.history.pop(0)
                
            return result
                
        except Exception as e:
            self.logger.error(f"预测注意力水平失败: {e}")
            return {
                'level': self.attention_levels[len(self.attention_levels) // 2],
                'level_idx': len(self.attention_levels) // 2,
                'probability': 0.0,
                'all_probs': {level: 0.0 for level in self.attention_levels}
            }
    
    def get_smoothed_attention(self):
        """
        获取平滑后的注意力水平
        通过对历史预测结果进行加权平均，减少波动
        
        返回:
            平滑后的注意力水平结果
        """
        if not self.history:
            return None
            
        # 计算平滑注意力
        weights = np.linspace(0.5, 1.0, len(self.history))  # 最近的预测权重更高
        weights = weights / np.sum(weights)  # 归一化权重
        
        # 计算加权平均概率
        avg_probs = {level: 0.0 for level in self.attention_levels}
        for i, hist in enumerate(self.history):
            for level, prob in hist['all_probs'].items():
                avg_probs[level] += prob * weights[i]
        
        # 找出概率最高的注意力水平
        max_level = max(avg_probs.items(), key=lambda x: x[1])
        level = max_level[0]
        probability = max_level[1]
        level_idx = self.attention_levels.index(level)
        
        return {
            'level': level,
            'level_idx': level_idx,
            'probability': probability,
            'all_probs': avg_probs
        }
    
    def attention_score_to_numeric(self, attention_result):
        """
        将注意力水平转换为数值分数(0-100)
        
        参数:
            attention_result: 注意力预测结果
            
        返回:
            注意力分数(0-100)
        """
        if attention_result is None:
            return 50  # 默认中等注意力
            
        # 获取注意力水平索引
        level_idx = attention_result['level_idx']
        
        # 计算基础分数 (将attention_levels的索引映射到0-100范围)
        base_score = (level_idx / (len(self.attention_levels) - 1)) * 100
        
        # 考虑概率因素进行调整
        probability = attention_result['probability']
        score = base_score * (0.8 + 0.2 * probability)  # 概率影响20%的分数
        
        # 确保分数在0-100范围内
        score = max(0, min(100, score))
        
        return round(score, 1)  # 精确到小数点后一位
    
    def evaluate_simulation_data(self):
        """
        对模拟数据进行评估，用于测试模型性能
        
        返回:
            评估结果
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练，无法进行模拟评估")
            return None
            
        # 生成模拟数据
        sim_data = self._generate_simulation_data()
        
        # 进行预测
        predictions = []
        for features, true_label in sim_data:
            pred = self.predict(features)
            predictions.append((true_label, pred['level'], pred['probability']))
        
        # 计算评估指标
        y_true = [p[0] for p in predictions]
        y_pred = [p[1] for p in predictions]
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.attention_levels)
        report = classification_report(y_true, y_pred, target_names=self.attention_levels)
        
        self.logger.info(f"模拟数据评估 - 精度: {accuracy:.4f}")
        self.logger.debug(f"混淆矩阵:\n{conf_matrix}")
        self.logger.debug(f"分类报告:\n{report}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }
    
    def _generate_simulation_data(self):
        """
        生成模拟特征数据用于评估
        
        返回:
            模拟数据列表 [(特征, 标签), ...]
        """
        simulation_data = []
        num_samples = 100
        
        # 为每个注意力水平生成模拟特征
        for level in self.attention_levels:
            level_idx = self.attention_levels.index(level)
            
            # 创建特征模板 - 这里使用简化模型，真实应用需要更复杂的特征模拟
            base_features = {
                'theta_power_ch0': 0.5 + 0.1 * level_idx,
                'alpha_power_ch0': 0.3 + 0.15 * level_idx,
                'beta_power_ch0': 0.2 + 0.2 * level_idx,
                'corr_TP9_AF7': 0.4 + 0.05 * level_idx,
                'wavelet_approx_energy_ch0': 0.6 + 0.1 * level_idx,
                'zero_crossing_rate_ch0': 0.3 - 0.03 * level_idx
            }
            
            # 为每个样本添加随机噪声
            for _ in range(num_samples // len(self.attention_levels)):
                noisy_features = base_features.copy()
                for key in noisy_features:
                    noisy_features[key] += np.random.normal(0, 0.05)
                
                simulation_data.append((noisy_features, level))
        
        # 打乱数据顺序
        np.random.shuffle(simulation_data)
        
        return simulation_data 