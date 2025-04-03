#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
认证相关API路由
"""

from flask import Blueprint, request, jsonify, current_app
import jwt
from datetime import datetime, timedelta
import os
import logging

# 创建蓝图
auth_bp = Blueprint('auth', __name__)

# 返回错误的辅助函数
def error_response(message, status_code=400):
    response = jsonify({'error': message})
    response.status_code = status_code
    return response

#========== 认证相关路由 ==========#

@auth_bp.route('/login', methods=['POST'])
def login():
    """用户登录接口"""
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return error_response('缺少用户名或密码', 400)
    
    username = data['username']
    password = data['password']
    
    # 简单的示例认证逻辑（实际应用中应使用更安全的方式）
    # 这里仅用于演示，实际项目应该连接到用户数据库进行认证
    valid_users = {
        'admin': 'admin123',
        'teacher': 'teacher123',
        'student': 'student123'
    }
    
    if username in valid_users and valid_users[username] == password:
        # 创建令牌
        token_expiration = current_app.config.get('TOKEN_EXPIRATION', 24 * 60 * 60)  # 默认24小时
        exp_time = datetime.now() + timedelta(seconds=token_expiration)
        token_payload = {
            'user_id': username,
            'exp': exp_time.timestamp(),
            'iat': datetime.now().timestamp(),
            'role': 'admin' if username == 'admin' else ('teacher' if username == 'teacher' else 'student')
        }
        token = jwt.encode(token_payload, current_app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'expires_at': exp_time.isoformat(),
            'user_id': username,
            'role': token_payload['role']
        })
    
    return error_response('无效的用户名或密码', 401)

@auth_bp.route('/validate', methods=['POST'])
def validate_token():
    """验证令牌有效性"""
    data = request.get_json()
    
    if not data or 'token' not in data:
        return error_response('缺少令牌', 400)
    
    token = data['token']
    
    try:
        # 解码token
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        
        # 检查token是否过期
        exp_time = datetime.fromtimestamp(payload['exp'])
        if exp_time < datetime.now():
            return jsonify({
                'valid': False,
                'message': '令牌已过期',
                'expires_at': exp_time.isoformat()
            })
            
        # 令牌有效
        return jsonify({
            'valid': True,
            'user_id': payload['user_id'],
            'role': payload.get('role', 'user'),
            'expires_at': exp_time.isoformat()
        })
    
    except jwt.ExpiredSignatureError:
        return jsonify({
            'valid': False,
            'message': '令牌已过期'
        })
    except jwt.InvalidTokenError as e:
        return jsonify({
            'valid': False,
            'message': f'无效的令牌: {str(e)}'
        })

@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """刷新访问令牌"""
    data = request.get_json()
    
    if not data or 'token' not in data:
        return error_response('缺少令牌', 400)
    
    token = data['token']
    
    try:
        # 解码旧token
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        
        # 创建新token
        token_expiration = current_app.config.get('TOKEN_EXPIRATION', 24 * 60 * 60)  # 默认24小时
        exp_time = datetime.now() + timedelta(seconds=token_expiration)
        new_payload = {
            'user_id': payload['user_id'],
            'exp': exp_time.timestamp(),
            'iat': datetime.now().timestamp(),
            'role': payload.get('role', 'user')
        }
        new_token = jwt.encode(new_payload, current_app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': new_token,
            'expires_at': exp_time.isoformat(),
            'user_id': payload['user_id'],
            'role': new_payload['role']
        })
    
    except jwt.ExpiredSignatureError:
        return error_response('令牌已过期，无法刷新', 401)
    except jwt.InvalidTokenError as e:
        return error_response(f'无效的令牌: {str(e)}', 401)

@auth_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'version': current_app.config.get('VERSION', '1.0.0')
    }) 