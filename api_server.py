#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API服务器入口脚本
集成REST和WebSocket服务器，提供HBN EEG数据访问接口
"""

import os
import sys
import logging
import argparse
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# 确保可以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入REST API服务器
from api.rest.server import create_app

# 导入WebSocket服务器的main函数
from api.websocket.server import main as ws_main

# 设置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_rest_server(host='0.0.0.0', port=5000, debug=False):
    """启动REST API服务器"""
    logger.info(f"正在启动REST API服务器，监听 {host}:{port}...")
    app = create_app()
    app.run(host=host, port=port, debug=debug, use_reloader=False)

async def start_websocket_server():
    """启动WebSocket服务器"""
    logger.info("正在启动WebSocket服务器...")
    await ws_main()

def run_ws_server_in_thread():
    """在单独的线程中运行WebSocket服务器"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="脑注意力康复系统API服务器")
    parser.add_argument("--rest-host", default="0.0.0.0", help="REST API服务器主机名")
    parser.add_argument("--rest-port", type=int, default=5000, help="REST API服务器端口")
    parser.add_argument("--ws-host", default="0.0.0.0", help="WebSocket服务器主机名")
    parser.add_argument("--ws-port", type=int, default=5001, help="WebSocket服务器端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--only-rest", action="store_true", help="仅启动REST API服务器")
    parser.add_argument("--only-ws", action="store_true", help="仅启动WebSocket服务器")
    parser.add_argument("--data-dir", default="data/eeg/hbn", help="HBN EEG数据目录")
    args = parser.parse_args()

    # 设置环境变量
    os.environ['API_HOST'] = args.rest_host
    os.environ['API_PORT'] = str(args.rest_port)
    os.environ['API_DEBUG'] = str(args.debug).lower()
    os.environ['WS_HOST'] = args.ws_host
    os.environ['WS_PORT'] = str(args.ws_port)
    os.environ['DATA_DIR'] = args.data_dir

    # 根据参数启动服务器
    if args.only_rest:
        # 仅启动REST API服务器
        start_rest_server(args.rest_host, args.rest_port, args.debug)
    elif args.only_ws:
        # 仅启动WebSocket服务器
        asyncio.run(start_websocket_server())
    else:
        # 同时启动两个服务器
        logger.info("同时启动REST API和WebSocket服务器...")
        
        # 在单独的线程中启动WebSocket服务器
        ws_thread = threading.Thread(target=run_ws_server_in_thread)
        ws_thread.daemon = True
        ws_thread.start()
        
        # 在主线程中启动REST API服务器
        start_rest_server(args.rest_host, args.rest_port, args.debug)

if __name__ == "__main__":
    main() 