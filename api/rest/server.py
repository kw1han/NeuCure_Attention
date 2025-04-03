#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RESTful API服务器
提供HBN EEG数据的访问接口
"""

import os
import sys
import logging
from flask import Flask
from flask_cors import CORS

# 确保可以导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.hbn_eeg_loader import HBNEEGLoader
from api.rest.routes.auth_routes import auth_bp
from api.rest.routes.eeg_routes import eeg_bp

def create_app():
    """创建并配置Flask应用"""
    # 创建应用
    app = Flask(__name__)
    CORS(app)  # 允许跨域请求
    
    # 配置
    app.config['SECRET_KEY'] = os.environ.get('API_SECRET_KEY', 'brain-attention-system-secret')
    app.config['DATA_DIR'] = os.environ.get('DATA_DIR', 'data/eeg/hbn')
    app.config['PROCESSED_DIR'] = os.environ.get('PROCESSED_DIR', 'data/eeg/processed')
    app.config['TOKEN_EXPIRATION'] = int(os.environ.get('TOKEN_EXPIRATION', 24 * 60 * 60))  # 默认24小时
    app.config['VERSION'] = '1.0.0'
    
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app.logger.setLevel(logging.INFO)
    
    # 初始化EEG数据加载器
    eeg_loader = HBNEEGLoader(
        data_dir=app.config['DATA_DIR'],
        processed_dir=app.config['PROCESSED_DIR'],
        logger=app.logger
    )
    
    # 将加载器添加到应用配置
    app.config['EEG_LOADER'] = eeg_loader
    
    # 注册蓝图
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(eeg_bp, url_prefix='/api')
    
    # 扫描可用的EEG文件
    @app.before_first_request
    def before_first_request():
        app.logger.info("扫描可用的HBN EEG文件...")
        subjects = eeg_loader.get_subject_list()
        app.logger.info(f"找到 {len(subjects)} 个被试。")
    
    return app

# 启动服务器
if __name__ == "__main__":
    # 创建应用
    app = create_app()
    
    # 设置主机和端口
    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', 5000))
    
    # 启动模式
    debug = os.environ.get('API_DEBUG', 'false').lower() == 'true'
    
    # 启动服务器
    app.logger.info(f"API服务器正在启动，监听 {host}:{port}...")
    app.run(host=host, port=port, debug=debug) 