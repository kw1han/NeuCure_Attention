#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主窗口模块
应用程序的主界面，包含注意力监测、训练等组件
"""

import sys
import os
import time
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QLabel, QPushButton, QProgressBar, QMessageBox, QFileDialog,
    QComboBox, QSpinBox, QCheckBox, QGroupBox, QSplitter, QAction
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSettings
from PyQt5.QtGui import QIcon, QFont, QPixmap

from utils.attention_monitor import AttentionMonitor
from components.attention_dashboard import AttentionDashboard
from components.training_game import TrainingGameWidget
from components.settings_dialog import SettingsDialog
from components.report_viewer import ReportViewer

class MainWindow(QMainWindow):
    """应用程序主窗口"""
    
    def __init__(self, config, logger):
        """
        初始化主窗口
        
        参数:
            config: 配置信息字典
            logger: 日志记录器
        """
        super().__init__()
        
        self.config = config
        self.logger = logger
        
        # 创建注意力监测器
        self.attention_monitor = AttentionMonitor(config)
        
        # 设置窗口
        self.setWindowTitle(config.get('app_name', '儿童注意力康复训练系统'))
        self.setMinimumSize(1024, 768)
        self.setup_ui()
        
        # 设置信号连接
        self.setup_connections()
        
        # 监测状态
        self.monitoring_active = False
        self.calibration_active = False
        
        # 状态更新计时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # 每秒更新一次状态
        
        # 加载用户设置
        self.settings = QSettings("BrainAttentionSystem", "UserSettings")
        self.load_settings()
        
        self.logger.info("主窗口初始化完成")
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 添加注意力监测标签页
        self.setup_monitor_tab()
        
        # 添加训练游戏标签页
        self.setup_training_tab()
        
        # 添加报告标签页
        self.setup_report_tab()
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 状态栏标签
        self.connection_status_label = QLabel("状态: 未连接")
        self.statusBar().addPermanentWidget(self.connection_status_label)
        
        self.device_info_label = QLabel()
        self.statusBar().addPermanentWidget(self.device_info_label)
        
        # 设置菜单
        self.setup_menu()
    
    def setup_menu(self):
        """设置菜单栏"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        
        export_action = QAction("导出数据", self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 设备菜单
        device_menu = self.menuBar().addMenu("设备")
        
        connect_action = QAction("连接设备", self)
        connect_action.triggered.connect(self.toggle_monitoring)
        self.connect_action = connect_action
        device_menu.addAction(connect_action)
        
        calibrate_action = QAction("校准", self)
        calibrate_action.triggered.connect(self.start_calibration)
        calibrate_action.setEnabled(False)
        self.calibrate_action = calibrate_action
        device_menu.addAction(calibrate_action)
        
        # 设置菜单
        settings_menu = self.menuBar().addMenu("设置")
        
        app_settings_action = QAction("应用设置", self)
        app_settings_action.triggered.connect(self.show_settings)
        settings_menu.addAction(app_settings_action)
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_action = QAction("帮助", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def setup_monitor_tab(self):
        """设置注意力监测标签页"""
        monitor_tab = QWidget()
        layout = QVBoxLayout(monitor_tab)
        
        # 控制区域
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)
        
        # 监测控制按钮
        self.start_button = QPushButton("开始监测")
        self.start_button.setMinimumWidth(120)
        self.start_button.clicked.connect(self.toggle_monitoring)
        control_layout.addWidget(self.start_button)
        
        # 校准按钮
        self.calibrate_button = QPushButton("校准注意力")
        self.calibrate_button.setMinimumWidth(120)
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.calibrate_button.setEnabled(False)
        control_layout.addWidget(self.calibrate_button)
        
        # 校准进度条
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setVisible(False)
        control_layout.addWidget(self.calibration_progress)
        
        control_layout.addStretch()
        
        # 注意力面板
        self.attention_dashboard = AttentionDashboard()
        layout.addWidget(self.attention_dashboard)
        
        self.tab_widget.addTab(monitor_tab, "注意力监测")
    
    def setup_training_tab(self):
        """设置训练游戏标签页"""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # 游戏选择区域
        game_selection_layout = QHBoxLayout()
        layout.addLayout(game_selection_layout)
        
        game_selection_layout.addWidget(QLabel("选择训练游戏:"))
        
        self.game_combo = QComboBox()
        self.game_combo.addItem("太空宝贝", "space_baby")
        self.game_combo.addItem("魔法森林大冒险", "magic_forest")
        self.game_combo.addItem("色彩拼图奇遇", "color_puzzle")
        game_selection_layout.addWidget(self.game_combo)
        
        game_selection_layout.addWidget(QLabel("难度:"))
        
        self.difficulty_combo = QComboBox()
        for i in range(1, 6):
            self.difficulty_combo.addItem(f"{i}级", i)
        game_selection_layout.addWidget(self.difficulty_combo)
        
        self.auto_difficulty_check = QCheckBox("自动调整难度")
        self.auto_difficulty_check.setChecked(True)
        game_selection_layout.addWidget(self.auto_difficulty_check)
        
        game_selection_layout.addWidget(QLabel("训练时长(分钟):"))
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 60)
        self.duration_spin.setValue(10)
        game_selection_layout.addWidget(self.duration_spin)
        
        self.start_game_button = QPushButton("开始训练")
        self.start_game_button.clicked.connect(self.start_training_game)
        self.start_game_button.setEnabled(False)
        game_selection_layout.addWidget(self.start_game_button)
        
        game_selection_layout.addStretch()
        
        # 游戏区域
        self.training_game_widget = TrainingGameWidget(self.config)
        layout.addWidget(self.training_game_widget)
        
        self.tab_widget.addTab(training_tab, "训练游戏")
    
    def setup_report_tab(self):
        """设置报告标签页"""
        self.report_viewer = ReportViewer(self.config)
        self.tab_widget.addTab(self.report_viewer, "训练报告")
    
    def setup_connections(self):
        """设置信号连接"""
        # 注意力监测器信号
        self.attention_monitor.attention_updated.connect(self.update_attention)
        self.attention_monitor.raw_eeg_updated.connect(self.update_eeg_data)
        self.attention_monitor.connection_status_changed.connect(self.update_connection_status)
        self.attention_monitor.calibration_progress.connect(self.update_calibration_progress)
        self.attention_monitor.calibration_complete.connect(self.handle_calibration_complete)
        self.attention_monitor.error_occurred.connect(self.handle_error)
        
        # 训练游戏信号
        self.training_game_widget.game_completed.connect(self.handle_game_complete)
    
    def load_settings(self):
        """加载用户设置"""
        # 窗口尺寸和位置
        if self.settings.contains("window/geometry"):
            self.restoreGeometry(self.settings.value("window/geometry"))
        if self.settings.contains("window/state"):
            self.restoreState(self.settings.value("window/state"))
            
        # 游戏设置
        if self.settings.contains("game/type"):
            index = self.game_combo.findData(self.settings.value("game/type"))
            if index >= 0:
                self.game_combo.setCurrentIndex(index)
                
        if self.settings.contains("game/difficulty"):
            self.difficulty_combo.setCurrentIndex(int(self.settings.value("game/difficulty", 0)))
            
        if self.settings.contains("game/auto_difficulty"):
            self.auto_difficulty_check.setChecked(bool(self.settings.value("game/auto_difficulty", True)))
            
        if self.settings.contains("game/duration"):
            self.duration_spin.setValue(int(self.settings.value("game/duration", 10)))
    
    def save_settings(self):
        """保存用户设置"""
        # 窗口尺寸和位置
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        
        # 游戏设置
        self.settings.setValue("game/type", self.game_combo.currentData())
        self.settings.setValue("game/difficulty", self.difficulty_combo.currentIndex())
        self.settings.setValue("game/auto_difficulty", self.auto_difficulty_check.isChecked())
        self.settings.setValue("game/duration", self.duration_spin.value())
    
    @pyqtSlot()
    def toggle_monitoring(self):
        """切换注意力监测状态"""
        if not self.monitoring_active:
            # 启动监测
            if self.attention_monitor.start():
                self.monitoring_active = True
                self.start_button.setText("停止监测")
                self.calibrate_button.setEnabled(True)
                self.start_game_button.setEnabled(True)
                self.connect_action.setText("断开连接")
                self.calibrate_action.setEnabled(True)
                self.logger.info("注意力监测已启动")
        else:
            # 停止监测
            if self.attention_monitor.stop():
                self.monitoring_active = False
                self.start_button.setText("开始监测")
                self.calibrate_button.setEnabled(False)
                self.start_game_button.setEnabled(False)
                self.connect_action.setText("连接设备")
                self.calibrate_action.setEnabled(False)
                self.logger.info("注意力监测已停止")
    
    @pyqtSlot()
    def start_calibration(self):
        """开始注意力校准"""
        if not self.monitoring_active:
            QMessageBox.warning(self, "警告", "请先开始注意力监测")
            return
            
        if self.calibration_active:
            QMessageBox.information(self, "提示", "校准已在进行中")
            return
            
        # 显示提示对话框
        response = QMessageBox.question(
            self, 
            "注意力校准", 
            f"校准过程大约需要{self.config['signal_processing']['baseline_duration']}秒。\n"
            "在此期间，请保持放松状态，视线平视前方。\n\n"
            "准备好后点击"是"开始校准。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            # 开始校准
            if self.attention_monitor.start_calibration():
                self.calibration_active = True
                self.calibration_progress.setValue(0)
                self.calibration_progress.setVisible(True)
                self.start_button.setEnabled(False)
                self.calibrate_button.setEnabled(False)
                self.start_game_button.setEnabled(False)
                self.logger.info("注意力校准已开始")
            else:
                QMessageBox.warning(self, "警告", "无法启动校准")
    
    @pyqtSlot(dict)
    def update_attention(self, attention_data):
        """
        更新注意力数据
        
        参数:
            attention_data: 注意力数据字典
        """
        # 更新注意力面板
        self.attention_dashboard.update_attention(attention_data)
        
        # 更新训练游戏
        if self.training_game_widget.is_game_running():
            self.training_game_widget.update_attention(attention_data)
    
    @pyqtSlot(np.ndarray)
    def update_eeg_data(self, eeg_data):
        """
        更新EEG数据
        
        参数:
            eeg_data: EEG数据数组
        """
        # 更新注意力面板的EEG显示
        self.attention_dashboard.update_eeg_data(eeg_data)
    
    @pyqtSlot(bool, str)
    def update_connection_status(self, connected, message):
        """
        更新连接状态
        
        参数:
            connected: 是否已连接
            message: 状态消息
        """
        # 更新状态栏
        self.connection_status_label.setText(f"状态: {message}")
        
        # 设置颜色
        if connected:
            self.connection_status_label.setStyleSheet("color: green")
        else:
            self.connection_status_label.setStyleSheet("color: red")
    
    @pyqtSlot(int)
    def update_calibration_progress(self, progress):
        """
        更新校准进度
        
        参数:
            progress: 进度百分比(0-100)
        """
        self.calibration_progress.setValue(progress)
    
    @pyqtSlot(bool, str)
    def handle_calibration_complete(self, success, message):
        """
        处理校准完成事件
        
        参数:
            success: 校准是否成功
            message: 结果消息
        """
        self.calibration_active = False
        self.calibration_progress.setVisible(False)
        self.start_button.setEnabled(True)
        self.calibrate_button.setEnabled(True)
        self.start_game_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "校准完成", message)
        else:
            QMessageBox.warning(self, "校准失败", message)
            
        self.logger.info(f"注意力校准完成: {message}")
    
    @pyqtSlot(str)
    def handle_error(self, error_message):
        """
        处理错误事件
        
        参数:
            error_message: 错误消息
        """
        QMessageBox.critical(self, "错误", error_message)
        self.logger.error(f"系统错误: {error_message}")
    
    @pyqtSlot()
    def update_status(self):
        """更新状态信息"""
        if self.monitoring_active:
            # 获取设备信息
            device_info = self.attention_monitor.eeg_acquisition.get_device_info()
            
            # 计算运行时间
            if 'elapsed_time' in device_info:
                elapsed = device_info['elapsed_time']
                hours = int(elapsed / 3600)
                minutes = int((elapsed % 3600) / 60)
                seconds = int(elapsed % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # 有效采样率
                sample_rate = device_info.get('effective_sample_rate', 0)
                
                self.device_info_label.setText(
                    f"运行时间: {time_str} | "
                    f"采样率: {sample_rate:.1f} Hz | "
                    f"样本数: {device_info['total_samples']}"
                )
            else:
                self.device_info_label.setText("")
        else:
            self.device_info_label.setText("")
    
    @pyqtSlot()
    def start_training_game(self):
        """启动训练游戏"""
        if not self.monitoring_active:
            QMessageBox.warning(self, "警告", "请先开始注意力监测")
            return
            
        # 获取游戏设置
        game_type = self.game_combo.currentData()
        difficulty = self.difficulty_combo.currentIndex() + 1
        auto_adjust = self.auto_difficulty_check.isChecked()
        duration = self.duration_spin.value()
        
        # 启动游戏
        game_config = {
            'game_type': game_type,
            'difficulty': difficulty,
            'auto_adjust': auto_adjust,
            'duration': duration
        }
        
        if self.training_game_widget.start_game(game_config):
            self.tab_widget.setCurrentIndex(1)  # 切换到游戏标签页
            self.logger.info(f"已启动训练游戏: {game_type}, 难度: {difficulty}, 时长: {duration}分钟")
        else:
            QMessageBox.warning(self, "警告", "无法启动训练游戏")
    
    @pyqtSlot(dict)
    def handle_game_complete(self, game_results):
        """
        处理游戏完成事件
        
        参数:
            game_results: 游戏结果数据
        """
        # 显示游戏完成消息
        QMessageBox.information(
            self, 
            "训练完成", 
            f"恭喜完成训练！\n"
            f"游戏类型: {game_results['game_name']}\n"
            f"训练时长: {game_results['duration']/60:.1f}分钟\n"
            f"平均注意力分数: {game_results['avg_attention']:.1f}\n"
            f"最高注意力分数: {game_results['max_attention']:.1f}"
        )
        
        # 更新报告页面
        self.report_viewer.add_training_result(game_results)
        
        # 切换到报告标签页
        self.tab_widget.setCurrentIndex(2)
        
        self.logger.info(f"训练游戏完成: {game_results['game_name']}, 平均注意力: {game_results['avg_attention']:.1f}")
    
    @pyqtSlot()
    def export_data(self):
        """导出会话数据"""
        if not self.attention_monitor.get_attention_history():
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return
            
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "导出注意力数据", 
            f"attention_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )
        
        if file_path:
            # 导出数据
            if self.attention_monitor.export_session_data(file_path):
                QMessageBox.information(self, "成功", f"数据已导出到: {file_path}")
            else:
                QMessageBox.warning(self, "警告", "数据导出失败")
    
    @pyqtSlot()
    def show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_():
            # 应用新的设置
            updated_config = dialog.get_config()
            # 更新配置
            # 注意：某些设置可能需要重启应用才能生效
            self.logger.info("已更新应用设置")
    
    @pyqtSlot()
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, 
            "关于",
            f"<h3>儿童注意力康复训练系统</h3>"
            f"<p>版本: {self.config.get('version', '1.0.0')}</p>"
            f"<p>基于非侵入式脑机接口技术的儿童注意力康复训练系统，"
            f"通过实时监测脑电波信号，结合趣味游戏互动，帮助儿童提升注意力水平。</p>"
            f"<p>&copy; 2023 All Rights Reserved</p>"
        )
    
    @pyqtSlot()
    def show_help(self):
        """显示帮助对话框"""
        QMessageBox.information(
            self,
            "使用帮助",
            "<h3>快速使用指南</h3>"
            "<ol>"
            "<li>点击"开始监测"连接脑电设备</li>"
            "<li>进行"校准注意力"以获得个人基线数据</li>"
            "<li>在"训练游戏"标签页选择游戏类型和难度</li>"
            "<li>点击"开始训练"进行注意力训练</li>"
            "<li>训练完成后在"训练报告"标签页查看结果</li>"
            "</ol>"
            "<p>详细使用指南请参考用户手册。</p>"
        )
    
    def closeEvent(self, event):
        """
        窗口关闭事件
        
        参数:
            event: 关闭事件
        """
        # 保存设置
        self.save_settings()
        
        # 停止监测
        if self.monitoring_active:
            self.attention_monitor.stop()
            
        # 接受关闭事件
        event.accept() 