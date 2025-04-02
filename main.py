#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于非侵入式脑机接口的儿童注意力康复训练系统
主程序入口
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication

# 导入自定义模块
from components.main_window import MainWindow
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='儿童注意力康复训练系统')
    parser.add_argument('--config', type=str, default='config/default.json',
                        help='配置文件路径')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式')
    parser.add_argument('--simulate', action='store_true',
                        help='使用模拟数据替代实际脑电设备')
    return parser.parse_args()

def main():
    """主函数入口"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_level, log_file)
    
    logger.info("启动儿童注意力康复训练系统")
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # 设置是否使用模拟数据
    config['use_simulated_data'] = args.simulate
    
    # 启动 GUI 应用
    app = QApplication(sys.argv)
    app.setApplicationName("儿童注意力康复训练系统")
    app.setStyle("Fusion")  # 设置一个通用风格，确保在不同平台上有一致的外观
    
    # 创建主窗口
    window = MainWindow(config, logger)
    window.show()
    
    # 执行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 