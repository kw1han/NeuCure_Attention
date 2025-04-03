#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG数据相关API路由
"""

from flask import Blueprint, request, jsonify, current_app
import os
import sys
import numpy as np
import logging
from functools import wraps
import jwt
from datetime import datetime

# 创建蓝图
eeg_bp = Blueprint('eeg', __name__)

# 认证装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # 从请求头获取token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': '缺少认证令牌！'}), 401
        
        try:
            # 解码token
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = payload['user_id']
            
            # 检查token是否过期
            if datetime.fromtimestamp(payload['exp']) < datetime.now():
                return jsonify({'message': '令牌已过期！'}), 401
                
        except jwt.InvalidTokenError:
            return jsonify({'message': '无效的令牌！'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# 返回错误的辅助函数
def error_response(message, status_code=400):
    response = jsonify({'error': message})
    response.status_code = status_code
    return response

#========== EEG数据相关路由 ==========#

@eeg_bp.route('/subjects', methods=['GET'])
@token_required
def get_subjects(current_user):
    """获取所有被试ID"""
    eeg_loader = current_app.config['EEG_LOADER']
    
    subjects = eeg_loader.get_subject_list()
    
    # 如果没有找到被试，可能需要重新扫描
    if not subjects:
        eeg_loader.scan_available_files()
        subjects = eeg_loader.get_subject_list()
    
    return jsonify({
        'subjects': subjects,
        'count': len(subjects)
    })

@eeg_bp.route('/subjects/<subject_id>', methods=['GET'])
@token_required
def get_subject_data(current_user, subject_id):
    """获取特定被试的数据"""
    eeg_loader = current_app.config['EEG_LOADER']
    preprocess = request.args.get('preprocess', 'true').lower() == 'true'
    
    data_dict = eeg_loader.load_subject_data(subject_id, preprocess=preprocess)
    
    if data_dict is None:
        return error_response(f"找不到被试 {subject_id} 的数据", 404)
    
    # 从MNE Raw对象中提取基本信息
    raw = data_dict['raw']
    
    # 提取基本信息
    info = {
        'subject_id': subject_id,
        'sample_rate': raw.info['sfreq'],
        'channel_count': len(raw.ch_names),
        'channels': raw.ch_names,
        'duration': raw.times[-1],  # 数据时长（秒）
        'file_info': data_dict['file_info']
    }
    
    # 添加特征信息
    features = data_dict['features']
    
    # 计算注意力分数
    attention_score = eeg_loader.get_attention_score(features)
    
    # 组织返回数据
    result = {
        'info': info,
        'features': features,
        'attention_score': attention_score
    }
    
    return jsonify(result)

@eeg_bp.route('/subjects/<subject_id>/raw', methods=['GET'])
@token_required
def get_subject_raw_data(current_user, subject_id):
    """获取特定被试的原始EEG数据（用于可视化）"""
    eeg_loader = current_app.config['EEG_LOADER']
    preprocess = request.args.get('preprocess', 'true').lower() == 'true'
    start = float(request.args.get('start', 0))  # 开始时间（秒）
    duration = float(request.args.get('duration', 10))  # 持续时间（秒）
    
    data_dict = eeg_loader.load_subject_data(subject_id, preprocess=preprocess)
    
    if data_dict is None:
        return error_response(f"找不到被试 {subject_id} 的数据", 404)
    
    # 从MNE Raw对象中提取原始数据
    raw = data_dict['raw']
    sfreq = raw.info['sfreq']
    
    # 计算采样点范围
    start_sample = int(start * sfreq)
    end_sample = int((start + duration) * sfreq)
    
    # 获取数据
    data, times = raw[:, start_sample:end_sample]
    
    # 组织返回数据
    result = {
        'subject_id': subject_id,
        'sample_rate': sfreq,
        'channel_names': raw.ch_names,
        'times': times.tolist(),
        'data': data.tolist(),
        'start_time': start,
        'duration': duration
    }
    
    return jsonify(result)

@eeg_bp.route('/simulate', methods=['POST'])
@token_required
def simulate_eeg_data(current_user):
    """生成模拟的EEG数据"""
    eeg_loader = current_app.config['EEG_LOADER']
    data = request.get_json()
    
    # 获取参数，使用默认值
    seconds = data.get('seconds', 60)
    sfreq = data.get('sample_rate', 256)
    attention_level = data.get('attention_level')  # None表示随机变化
    
    # 生成模拟数据
    raw, features = eeg_loader.generate_simulated_data(
        seconds=seconds,
        sfreq=sfreq,
        attention_level=attention_level
    )
    
    # 提取原始数据
    data, times = raw[:, :]
    
    # 计算注意力分数
    attention_score = eeg_loader.get_attention_score(features)
    
    # 组织返回数据
    result = {
        'sample_rate': sfreq,
        'channel_names': raw.ch_names,
        'times': times.tolist(),
        'data': data.tolist(),
        'features': features,
        'attention_score': attention_score,
        'duration': seconds
    }
    
    return jsonify(result)

@eeg_bp.route('/attention/predict', methods=['POST'])
@token_required
def predict_attention(current_user):
    """从原始EEG数据预测注意力水平"""
    eeg_loader = current_app.config['EEG_LOADER']
    data = request.get_json()
    
    if not data or 'eeg_data' not in data or 'channels' not in data or 'sample_rate' not in data:
        return error_response('缺少必要的EEG数据信息', 400)
    
    # 提取数据
    eeg_data = data['eeg_data']  # 应为二维数组 [channels, samples]
    channels = data['channels']  # 通道名称列表
    sfreq = data['sample_rate']  # 采样率
    
    try:
        # 创建MNE Raw对象
        import numpy as np
        import mne
        
        # 确保数据是numpy数组
        eeg_data_np = np.array(eeg_data)
        
        # 创建MNE Info对象
        info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types='eeg')
        
        # 创建Raw对象
        raw = mne.io.RawArray(eeg_data_np, info)
        
        # 预处理
        raw = eeg_loader.preprocess_raw(raw)
        
        # 提取特征
        features = eeg_loader.extract_attention_features(raw)
        
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
        
        return jsonify({
            'attention_score': attention_score,
            'attention_level': level,
            'features': features
        })
        
    except Exception as e:
        current_app.logger.error(f"预测注意力时出错: {str(e)}")
        return error_response(f"处理数据时出错: {str(e)}", 500) 