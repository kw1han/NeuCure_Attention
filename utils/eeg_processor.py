#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG信号处理模块
负责脑电信号的预处理、特征提取和分析
"""

import numpy as np
import pandas as pd
from scipy import signal
import pywt
import logging
from sklearn.preprocessing import StandardScaler

class EEGProcessor:
    """脑电信号处理类"""
    
    def __init__(self, config):
        """
        初始化EEG处理器
        
        参数:
            config: 配置信息字典
        """
        self.logger = logging.getLogger('attention_system')
        self.config = config
        
        # 从配置中获取参数
        self.sample_rate = config["device"]["sample_rate"]
        self.channels = config["device"]["channels"]
        self.channel_names = config["device"]["channel_names"]
        self.wavelet = config["signal_processing"]["wavelet"]
        self.decomp_level = config["signal_processing"]["decomposition_level"]
        self.frequency_bands = config["signal_processing"]["frequency_bands"]
        self.window_size = int(config["signal_processing"]["window_size"] * self.sample_rate)
        self.overlap = float(config["signal_processing"]["overlap"])
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        
        # 创建带通滤波器
        self._create_filters()
        
        self.logger.info("EEG处理器初始化完成")
    
    def _create_filters(self):
        """创建各频段的滤波器"""
        self.filters = {}
        nyquist = self.sample_rate / 2.0
        
        for band_name, band_range in self.frequency_bands.items():
            low, high = band_range
            low_norm = low / nyquist
            high_norm = high / nyquist
            
            # 创建带通滤波器
            b, a = signal.butter(4, [low_norm, high_norm], btype='bandpass')
            self.filters[band_name] = (b, a)
            
        # 创建陷波滤波器去除电源噪声(50/60Hz)
        notch_freq = 50.0  # 根据国家/地区可能需要调整为60Hz
        notch_width = 5.0
        
        b_notch, a_notch = signal.iirnotch(notch_freq / nyquist, 30, self.sample_rate)
        self.filters['notch'] = (b_notch, a_notch)
        
        self.logger.debug(f"已创建{len(self.filters)}个频段滤波器")
    
    def preprocess(self, eeg_data):
        """
        预处理原始EEG数据
        
        参数:
            eeg_data: 原始EEG数据, 形状为 (samples, channels)
            
        返回:
            预处理后的EEG数据
        """
        if eeg_data.shape[1] != self.channels:
            self.logger.error(f"数据通道数不匹配. 预期{self.channels}, 实际{eeg_data.shape[1]}")
            raise ValueError(f"数据通道数不匹配. 预期{self.channels}, 实际{eeg_data.shape[1]}")
        
        # 检查数据中是否有NaN或inf
        if np.isnan(eeg_data).any() or np.isinf(eeg_data).any():
            self.logger.warning("输入数据包含NaN或Inf值, 已替换为0")
            eeg_data = np.nan_to_num(eeg_data)
        
        # 去除基线漂移
        eeg_data = self._remove_baseline(eeg_data)
        
        # 应用陷波滤波器去除电源噪声
        b_notch, a_notch = self.filters['notch']
        for ch in range(self.channels):
            eeg_data[:, ch] = signal.filtfilt(b_notch, a_notch, eeg_data[:, ch])
        
        # 数据标准化
        eeg_data = self._normalize_data(eeg_data)
        
        return eeg_data
    
    def _remove_baseline(self, eeg_data):
        """去除基线漂移"""
        # 使用高通滤波器去除低频漂移
        nyquist = self.sample_rate / 2.0
        high_pass_freq = 0.5 / nyquist  # 0.5Hz高通滤波
        b, a = signal.butter(4, high_pass_freq, btype='highpass')
        
        for ch in range(self.channels):
            eeg_data[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
            
        return eeg_data
    
    def _normalize_data(self, eeg_data):
        """数据标准化"""
        # 对每个通道进行标准化
        for ch in range(self.channels):
            eeg_data[:, ch] = self.scaler.fit_transform(eeg_data[:, ch].reshape(-1, 1)).ravel()
            
        return eeg_data
    
    def extract_features(self, eeg_data):
        """
        从预处理后的EEG数据中提取特征
        
        参数:
            eeg_data: 预处理后的EEG数据
            
        返回:
            特征字典
        """
        # 特征集合
        features = {}
        
        # 提取频段能量特征
        band_powers = self._extract_band_powers(eeg_data)
        for band, powers in band_powers.items():
            for ch in range(self.channels):
                features[f"{band}_power_ch{ch}"] = powers[ch]
        
        # 提取相关性特征
        correlation_features = self._extract_correlation_features(eeg_data)
        features.update(correlation_features)
        
        # 提取小波变换特征
        wavelet_features = self._extract_wavelet_features(eeg_data)
        features.update(wavelet_features)
        
        # 提取统计特征
        stat_features = self._extract_statistical_features(eeg_data)
        features.update(stat_features)
        
        return features
    
    def _extract_band_powers(self, eeg_data):
        """提取各频段能量"""
        band_powers = {}
        
        for band_name, (b, a) in self.filters.items():
            if band_name == 'notch':
                continue
                
            # 应用带通滤波器
            filtered_data = np.zeros_like(eeg_data)
            for ch in range(self.channels):
                filtered_data[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
            
            # 计算功率谱密度
            powers = np.zeros(self.channels)
            for ch in range(self.channels):
                f, psd = signal.welch(filtered_data[:, ch], fs=self.sample_rate, nperseg=self.window_size)
                powers[ch] = np.sum(psd)
            
            band_powers[band_name] = powers
            
        return band_powers
    
    def _extract_correlation_features(self, eeg_data):
        """提取通道间相关性特征"""
        features = {}
        
        # 计算通道间的相关系数矩阵
        corr_matrix = np.corrcoef(eeg_data.T)
        
        # 选择上三角矩阵的元素作为特征（排除对角线）
        counter = 0
        for i in range(self.channels):
            for j in range(i+1, self.channels):
                ch1 = self.channel_names[i]
                ch2 = self.channel_names[j]
                features[f"corr_{ch1}_{ch2}"] = corr_matrix[i, j]
                counter += 1
                
        self.logger.debug(f"已提取{counter}个通道相关性特征")
        return features
    
    def _extract_wavelet_features(self, eeg_data):
        """提取小波变换特征"""
        features = {}
        
        for ch in range(self.channels):
            # 执行离散小波变换
            coeffs = pywt.wavedec(eeg_data[:, ch], self.wavelet, level=self.decomp_level)
            
            # 提取各级系数的统计特征
            for level, coef in enumerate(coeffs):
                # 对于每一级系数，计算其统计特征
                level_name = "approx" if level == 0 else f"detail{level}"
                
                features[f"wavelet_{level_name}_mean_ch{ch}"] = np.mean(coef)
                features[f"wavelet_{level_name}_std_ch{ch}"] = np.std(coef)
                features[f"wavelet_{level_name}_energy_ch{ch}"] = np.sum(coef**2)
                features[f"wavelet_{level_name}_entropy_ch{ch}"] = self._calculate_entropy(coef)
        
        return features
    
    def _extract_statistical_features(self, eeg_data):
        """提取统计特征"""
        features = {}
        
        for ch in range(self.channels):
            channel_data = eeg_data[:, ch]
            
            # 基本统计量
            features[f"mean_ch{ch}"] = np.mean(channel_data)
            features[f"std_ch{ch}"] = np.std(channel_data)
            features[f"var_ch{ch}"] = np.var(channel_data)
            features[f"kurtosis_ch{ch}"] = self._calculate_kurtosis(channel_data)
            features[f"skewness_ch{ch}"] = self._calculate_skewness(channel_data)
            
            # 零交叉率
            features[f"zero_crossing_rate_ch{ch}"] = self._calculate_zero_crossing_rate(channel_data)
            
            # 峰度和偏度
            features[f"peak_to_peak_ch{ch}"] = np.max(channel_data) - np.min(channel_data)
            
        return features
    
    def _calculate_entropy(self, data):
        """计算数据熵"""
        # 计算直方图
        hist, _ = np.histogram(data, bins=10)
        
        # 转换为概率
        hist = hist / np.sum(hist)
        
        # 去除零概率
        hist = hist[hist > 0]
        
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        # 零均值化
        data_zero_mean = data - np.mean(data)
        
        # 标准差
        std = np.std(data)
        if std == 0:
            return 0
            
        # 峰度计算
        n = len(data)
        kurtosis = (np.sum(data_zero_mean**4) / n) / (std**4) - 3
        
        return kurtosis
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        # 零均值化
        data_zero_mean = data - np.mean(data)
        
        # 标准差
        std = np.std(data)
        if std == 0:
            return 0
            
        # 偏度计算
        n = len(data)
        skewness = (np.sum(data_zero_mean**3) / n) / (std**3)
        
        return skewness
    
    def _calculate_zero_crossing_rate(self, data):
        """计算零交叉率"""
        # 信号符号变化的次数
        return np.sum(np.abs(np.diff(np.signbit(data)))) / len(data)
    
    def segment_data(self, eeg_data):
        """
        将数据分段用于特征提取
        
        参数:
            eeg_data: EEG数据
            
        返回:
            分段后的数据列表
        """
        segments = []
        step_size = int(self.window_size * (1 - self.overlap))
        
        for start in range(0, len(eeg_data) - self.window_size + 1, step_size):
            end = start + self.window_size
            segment = eeg_data[start:end, :]
            segments.append(segment)
            
        self.logger.debug(f"将数据分为{len(segments)}个片段")
        return segments
    
    def process_eeg_batch(self, eeg_data):
        """
        处理一批EEG数据，包括预处理、分段和特征提取
        
        参数:
            eeg_data: 原始EEG数据
            
        返回:
            特征列表
        """
        try:
            # 预处理
            preprocessed_data = self.preprocess(eeg_data)
            
            # 分段
            segments = self.segment_data(preprocessed_data)
            
            # 对每个片段提取特征
            all_features = []
            for segment in segments:
                features = self.extract_features(segment)
                all_features.append(features)
                
            self.logger.debug(f"已处理EEG数据批次，提取了{len(all_features)}组特征")
            return all_features
            
        except Exception as e:
            self.logger.error(f"处理EEG数据时出错: {e}")
            return [] 