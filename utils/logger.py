#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志设置模块
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(level=logging.INFO, log_file=None, max_bytes=10485760, backup_count=5):
    """
    设置并配置日志记录器
    
    参数:
        level: 日志级别 (默认: INFO)
        log_file: 日志文件路径 (默认: None，只输出到控制台)
        max_bytes: 日志文件最大字节数，超过后会轮转 (默认: 10MB)
        backup_count: 保留的历史日志文件数量 (默认: 5)
        
    返回:
        配置好的logger对象
    """
    # 创建logger
    logger = logging.getLogger('attention_system')
    logger.setLevel(level)
    
    # 清除之前的处理程序（如果有的话）
    if logger.handlers:
        logger.handlers.clear()
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理程序
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理程序
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 