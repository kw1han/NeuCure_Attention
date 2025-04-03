#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HBN EEG数据加载模块
提供HBN EEG数据集的加载、预处理和特征提取功能
"""

import os
import glob
import numpy as np
import pandas as pd
import mne
import pickle
from scipy import signal
from pathlib import Path
import logging
import warnings

# 忽略MNE的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

class HBNEEGLoader:
    """
    HBN EEG数据加载器
    负责加载、预处理HBN EEG数据，并提供模拟数据生成功能
    """
    
    def __init__(self, data_dir="data/eeg/hbn", processed_dir="data/eeg/processed", logger=None):
        """
        初始化加载器
        
        参数:
            data_dir: HBN数据目录
            processed_dir: 处理后数据保存目录
            logger: 日志记录器
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 确保目录存在
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # 缓存
        self.loaded_data = {}
        self.available_files = []
        
        # 扫描可用文件
        self.scan_available_files()
    
    def scan_available_files(self):
        """扫描可用的HBN EEG文件"""
        # 扫描EDF文件
        edf_files = glob.glob(os.path.join(self.data_dir, "**", "*.edf"), recursive=True)
        
        # 扫描BrainVision文件
        vhdr_files = glob.glob(os.path.join(self.data_dir, "**", "*.vhdr"), recursive=True)
        
        # 合并文件列表
        all_files = edf_files + vhdr_files
        
        # 提取基本信息
        file_info = []
        for file_path in all_files:
            try:
                # 提取相对路径
                rel_path = os.path.relpath(file_path, self.data_dir)
                
                # 尝试提取被试ID (假设文件名或父目录包含ID)
                parts = Path(file_path).parts
                subject_id = None
                for part in parts:
                    # 尝试寻找符合HBN命名格式的部分 (如NDARAA123ABC)
                    if part.startswith("NDAR") and len(part) >= 8:
                        subject_id = part
                        break
                
                # 如果没找到，使用文件名的一部分
                if not subject_id:
                    subject_id = os.path.splitext(os.path.basename(file_path))[0]
                
                # 文件格式
                file_format = os.path.splitext(file_path)[1].lower()
                
                file_info.append({
                    'path': file_path,
                    'rel_path': rel_path,
                    'subject_id': subject_id,
                    'format': file_format
                })
                
            except Exception as e:
                self.logger.warning(f"处理文件 {file_path} 时出错: {str(e)}")
        
        self.available_files = file_info
        self.logger.info(f"找到 {len(self.available_files)} 个HBN EEG文件")
        
        return file_info
    
    def get_subject_list(self):
        """获取可用的被试ID列表"""
        subjects = sorted(list(set(f['subject_id'] for f in self.available_files)))
        return subjects
    
    def load_eeg_file(self, file_path):
        """
        加载EEG文件
        
        参数:
            file_path: EEG文件路径
            
        返回:
            raw: MNE Raw对象
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.edf':
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            elif file_ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            return raw
            
        except Exception as e:
            self.logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            return None
    
    def preprocess_raw(self, raw, bandpass=(1, 50), notch=50, resample_rate=None):
        """
        预处理EEG数据
        
        参数:
            raw: MNE Raw对象
            bandpass: 带通滤波频率范围 (低, 高)
            notch: 陷波滤波频率 (电源干扰)
            resample_rate: 重采样频率 (Hz)
            
        返回:
            processed_raw: 处理后的MNE Raw对象
        """
        try:
            # 创建副本以避免修改原始数据
            raw_copy = raw.copy()
            
            # 应用带通滤波
            if bandpass:
                raw_copy.filter(bandpass[0], bandpass[1], verbose=False)
            
            # 应用陷波滤波 (电源干扰)
            if notch:
                raw_copy.notch_filter(notch, verbose=False)
            
            # 重采样
            if resample_rate:
                raw_copy.resample(resample_rate, verbose=False)
            
            return raw_copy
            
        except Exception as e:
            self.logger.error(f"预处理EEG数据失败: {str(e)}")
            return raw
    
    def extract_attention_features(self, raw):
        """
        从EEG数据中提取注意力相关特征
        
        参数:
            raw: MNE Raw对象
            
        返回:
            features: 特征字典
        """
        try:
            # 获取数据数组
            data = raw.get_data()
            ch_names = raw.ch_names
            sfreq = raw.info['sfreq']
            
            # 特征字典
            features = {}
            
            # 计算频带能量
            # Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            
            # 对每个通道计算频带能量
            band_powers = {}
            for band_name, freq_range in bands.items():
                band_powers[band_name] = {}
                
                # 使用FFT计算功率谱
                win_size = int(4 * sfreq)  # 4秒窗口
                step = int(sfreq)  # 1秒步长
                
                for ch_idx, ch_name in enumerate(ch_names):
                    ch_data = data[ch_idx]
                    
                    # 分窗计算
                    powers = []
                    for start in range(0, len(ch_data) - win_size, step):
                        segment = ch_data[start:start + win_size]
                        
                        # 加窗
                        segment = segment * np.hanning(len(segment))
                        
                        # 计算FFT
                        fft_vals = np.abs(np.fft.rfft(segment))
                        fft_freq = np.fft.rfftfreq(len(segment), 1.0/sfreq)
                        
                        # 计算频带能量
                        idx_band = np.logical_and(fft_freq >= freq_range[0], fft_freq <= freq_range[1])
                        band_power = np.sum(fft_vals[idx_band] ** 2)
                        powers.append(band_power)
                    
                    # 存储该通道的频带能量
                    band_powers[band_name][ch_name] = np.array(powers)
            
            # 计算注意力相关指标
            
            # 1. 前额叶theta/beta比值 (注意力指标)
            frontal_channels = [ch for ch in ch_names if ch.startswith(('F', 'Fp'))]
            if frontal_channels:
                theta_beta_ratio = []
                for ch in frontal_channels:
                    if ch in band_powers['theta'] and ch in band_powers['beta']:
                        ch_theta = band_powers['theta'][ch]
                        ch_beta = band_powers['beta'][ch]
                        ratio = np.mean(ch_theta) / np.mean(ch_beta)
                        theta_beta_ratio.append(ratio)
                
                features['frontal_theta_beta_ratio'] = np.mean(theta_beta_ratio) if theta_beta_ratio else None
            
            # 2. Alpha波中心频率 (注意力指标)
            alpha_center_freq = []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = data[ch_idx]
                
                # 计算功率谱
                freqs, psd = signal.welch(ch_data, sfreq, nperseg=int(sfreq * 2))
                
                # 提取alpha频段
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                alpha_freqs = freqs[alpha_mask]
                alpha_psd = psd[alpha_mask]
                
                # 计算中心频率 (加权平均)
                if len(alpha_psd) > 0 and np.sum(alpha_psd) > 0:
                    center_freq = np.sum(alpha_freqs * alpha_psd) / np.sum(alpha_psd)
                    alpha_center_freq.append(center_freq)
            
            features['alpha_center_frequency'] = np.mean(alpha_center_freq) if alpha_center_freq else None
            
            # 3. 皮层间相干性 (脑区连接性指标)
            # 简化版本 - 计算通道间alpha波段的相关性
            if len(ch_names) > 1:
                alpha_corr = []
                for i in range(len(ch_names)):
                    for j in range(i+1, len(ch_names)):
                        if ch_names[i] in band_powers['alpha'] and ch_names[j] in band_powers['alpha']:
                            ch1_alpha = band_powers['alpha'][ch_names[i]]
                            ch2_alpha = band_powers['alpha'][ch_names[j]]
                            
                            # 确保长度一致
                            min_len = min(len(ch1_alpha), len(ch2_alpha))
                            if min_len > 0:
                                corr = np.corrcoef(ch1_alpha[:min_len], ch2_alpha[:min_len])[0, 1]
                                alpha_corr.append(corr)
                
                features['alpha_coherence'] = np.mean(alpha_corr) if alpha_corr else None
            
            # 4. 频带能量比值
            # (theta+alpha)/(beta+gamma) 比值 - 放松vs集中注意力
            relaxation_ratio = []
            focus_ratio = []
            
            for ch_name in ch_names:
                if all(ch_name in band_powers[band] for band in bands.keys()):
                    theta = np.mean(band_powers['theta'][ch_name])
                    alpha = np.mean(band_powers['alpha'][ch_name])
                    beta = np.mean(band_powers['beta'][ch_name])
                    gamma = np.mean(band_powers['gamma'][ch_name])
                    
                    # 放松指标
                    if (beta + gamma) > 0:
                        relax = (theta + alpha) / (beta + gamma)
                        relaxation_ratio.append(relax)
                    
                    # 集中注意力指标
                    if theta > 0:
                        focus = beta / theta
                        focus_ratio.append(focus)
            
            features['relaxation_ratio'] = np.mean(relaxation_ratio) if relaxation_ratio else None
            features['focus_ratio'] = np.mean(focus_ratio) if focus_ratio else None
            
            # 5. 存储各频带平均能量
            for band in bands.keys():
                band_mean = []
                for ch in ch_names:
                    if ch in band_powers[band]:
                        band_mean.append(np.mean(band_powers[band][ch]))
                
                features[f'{band}_power'] = np.mean(band_mean) if band_mean else None
            
            return features
            
        except Exception as e:
            self.logger.error(f"提取注意力特征失败: {str(e)}")
            return {}
    
    def load_subject_data(self, subject_id, preprocess=True, cache=True):
        """
        加载特定被试的EEG数据
        
        参数:
            subject_id: 被试ID
            preprocess: 是否预处理
            cache: 是否使用缓存
            
        返回:
            data_dict: 数据字典，包含raw和features
        """
        # 检查缓存
        cache_key = f"{subject_id}_{preprocess}"
        if cache and cache_key in self.loaded_data:
            return self.loaded_data[cache_key]
        
        # 查找被试文件
        subject_files = [f for f in self.available_files if f['subject_id'] == subject_id]
        
        if not subject_files:
            self.logger.warning(f"未找到被试 {subject_id} 的数据文件")
            return None
        
        # 检查处理后的缓存文件
        cache_file = os.path.join(self.processed_dir, f"{subject_id}{'_processed' if preprocess else ''}.pkl")
        
        if cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data_dict = pickle.load(f)
                
                self.logger.info(f"从缓存加载被试 {subject_id} 的数据")
                
                # 更新缓存
                if cache:
                    self.loaded_data[cache_key] = data_dict
                
                return data_dict
            except Exception as e:
                self.logger.warning(f"加载缓存文件 {cache_file} 失败: {str(e)}")
        
        # 加载第一个可用文件
        file_path = subject_files[0]['path']
        raw = self.load_eeg_file(file_path)
        
        if raw is None:
            return None
        
        # 预处理
        if preprocess:
            raw = self.preprocess_raw(raw)
        
        # 提取特征
        features = self.extract_attention_features(raw)
        
        # 创建数据字典
        data_dict = {
            'raw': raw,
            'features': features,
            'subject_id': subject_id,
            'file_info': subject_files[0]
        }
        
        # 保存到缓存
        if cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_dict, f)
                self.logger.info(f"缓存被试 {subject_id} 的数据到 {cache_file}")
            except Exception as e:
                self.logger.warning(f"缓存数据到 {cache_file} 失败: {str(e)}")
            
            # 更新内存缓存
            self.loaded_data[cache_key] = data_dict
        
        return data_dict
    
    def generate_simulated_data(self, seconds=60, sfreq=256, attention_level=None):
        """
        生成模拟的EEG数据
        
        参数:
            seconds: 数据长度（秒）
            sfreq: 采样频率
            attention_level: 注意力水平 (0-100)，None表示随机变化
            
        返回:
            raw: MNE Raw对象
            features: 特征字典
        """
        # 定义通道
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        n_channels = len(ch_names)
        
        # 创建数据数组
        n_samples = int(seconds * sfreq)
        data = np.zeros((n_channels, n_samples))
        
        # 如果没有指定注意力水平，生成随机变化的注意力
        if attention_level is None:
            # 创建缓慢变化的注意力曲线
            base_attention = np.random.rand() * 70 + 30  # 30-100的基线
            attention_curve = np.zeros(n_samples)
            
            # 添加缓慢变化
            n_changes = int(seconds / 5)  # 平均每5秒变化一次
            change_points = np.sort(np.random.choice(n_samples, n_changes, replace=False))
            
            current_attention = base_attention
            last_point = 0
            for point in change_points:
                target_attention = max(10, min(100, current_attention + (np.random.rand() - 0.5) * 30))
                
                # 线性插值
                attention_curve[last_point:point] = np.linspace(current_attention, target_attention, point - last_point)
                
                current_attention = target_attention
                last_point = point
            
            # 最后一段
            attention_curve[last_point:] = current_attention
            
            # 归一化到0-1
            normalized_attention = attention_curve / 100.0
        else:
            # 固定注意力水平
            normalized_attention = np.ones(n_samples) * (attention_level / 100.0)
        
        # 频带定义
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # 基于注意力水平调整频带能量
        for ch_idx in range(n_channels):
            ch_data = np.zeros(n_samples)
            
            # 为每个频带生成信号
            for band_name, (low_freq, high_freq) in bands.items():
                # 频带内随机频率
                for freq in np.linspace(low_freq, high_freq, int((high_freq - low_freq) * 2)):
                    # 随机相位
                    phase = np.random.rand() * 2 * np.pi
                    
                    # 基础振幅
                    amp = 0.1 * np.random.rand() + 0.05
                    
                    # 根据注意力调整频带能量
                    band_amp = amp
                    
                    # 注意力高时beta, gamma增强，theta减弱
                    if band_name == 'beta' or band_name == 'gamma':
                        band_amp = amp * (0.5 + normalized_attention)
                    elif band_name == 'theta':
                        band_amp = amp * (1.5 - normalized_attention)
                    elif band_name == 'alpha':
                        # alpha在放松时增强，专注时减弱
                        # 创建更复杂的alpha模式 - 放松时(~0.5)最强
                        alpha_mod = 1.0 - 2.0 * np.abs(normalized_attention - 0.5)
                        band_amp = amp * (0.5 + alpha_mod)
                    
                    # 生成正弦波
                    t = np.arange(n_samples) / sfreq
                    wave = band_amp * np.sin(2 * np.pi * freq * t + phase)
                    
                    # 添加到通道数据
                    ch_data += wave
            
            # 添加噪声
            noise = np.random.normal(0, 0.05, n_samples)
            ch_data += noise
            
            # 存储到数据数组
            data[ch_idx] = ch_data
        
        # 创建MNE Raw对象
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # 提取特征
        features = self.extract_attention_features(raw)
        
        # 添加真实的注意力值
        features['true_attention'] = np.mean(normalized_attention) * 100
        
        return raw, features
    
    def get_attention_score(self, features):
        """
        从特征计算注意力分数
        
        参数:
            features: 特征字典
            
        返回:
            score: 注意力分数 (0-100)
        """
        # 如果有真实值直接返回
        if 'true_attention' in features:
            return features['true_attention']
        
        # 从特征估计注意力分数
        indicators = []
        
        # 1. 使用theta/beta比值 (反向指标)
        if 'frontal_theta_beta_ratio' in features and features['frontal_theta_beta_ratio'] is not None:
            # 典型范围1-3，值越小注意力越高
            ratio = features['frontal_theta_beta_ratio']
            if ratio > 0:
                # 将比值映射到0-100分
                score = max(0, min(100, 100 - (ratio - 1) * 30))
                indicators.append(score)
        
        # 2. 使用focus_ratio (正向指标)
        if 'focus_ratio' in features and features['focus_ratio'] is not None:
            ratio = features['focus_ratio']
            if ratio > 0:
                # 典型范围0.5-3，值越大注意力越高
                score = max(0, min(100, (ratio - 0.5) * 40))
                indicators.append(score)
        
        # 3. 使用alpha_center_frequency (正向指标)
        if 'alpha_center_frequency' in features and features['alpha_center_frequency'] is not None:
            freq = features['alpha_center_frequency']
            # 范围通常在8-13Hz，频率越高注意力越集中
            # 映射到注意力分数
            score = max(0, min(100, (freq - 8) * 20))
            indicators.append(score)
        
        # 4. 使用beta/theta功率比 (正向指标)
        if 'beta_power' in features and 'theta_power' in features:
            beta = features['beta_power']
            theta = features['theta_power']
            if theta > 0 and beta is not None and theta is not None:
                ratio = beta / theta
                # 映射到注意力分数
                score = max(0, min(100, ratio * 25))
                indicators.append(score)
        
        # 计算平均分数
        if indicators:
            return np.mean(indicators)
        else:
            # 没有有效指标，返回默认值
            return 50.0 