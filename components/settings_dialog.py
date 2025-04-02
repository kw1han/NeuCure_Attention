#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
设置对话框组件
提供系统配置和个人信息设置界面
"""

import os
import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, 
    QPushButton, QFileDialog, QFormLayout, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QSettings, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

class SettingsDialog(QDialog):
    """设置对话框"""
    
    # 定义信号
    settings_changed = pyqtSignal(dict)  # 设置更改信号
    
    def __init__(self, config, parent=None):
        """
        初始化设置对话框
        
        参数:
            config: 配置信息
            parent: 父窗口
        """
        super().__init__(parent)
        
        self.config = config.copy()
        self.original_config = config.copy()
        
        self.setWindowTitle("系统设置")
        self.setMinimumSize(600, 450)
        
        # 初始化界面
        self.setup_ui()
        
        # 加载设置
        self.load_settings()
    
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 添加一般设置标签页
        self.setup_general_tab()
        
        # 添加设备设置标签页
        self.setup_device_tab()
        
        # 添加用户信息标签页
        self.setup_user_tab()
        
        # 添加训练设置标签页
        self.setup_training_tab()
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 重置按钮
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self.reset_settings)
        button_layout.addWidget(reset_button)
        
        button_layout.addStretch()
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # 保存按钮
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.accept)
        save_button.setDefault(True)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
    
    def setup_general_tab(self):
        """设置一般设置标签页"""
        general_tab = QWidget()
        layout = QFormLayout(general_tab)
        
        # 应用名称
        self.app_name_edit = QLineEdit()
        layout.addRow("应用名称:", self.app_name_edit)
        
        # 日志级别
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItem("调试", "DEBUG")
        self.log_level_combo.addItem("信息", "INFO")
        self.log_level_combo.addItem("警告", "WARNING")
        self.log_level_combo.addItem("错误", "ERROR")
        layout.addRow("日志级别:", self.log_level_combo)
        
        # 自动保存
        self.auto_save_check = QCheckBox("启用")
        layout.addRow("自动保存数据:", self.auto_save_check)
        
        # 数据目录
        data_dir_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setReadOnly(True)
        data_dir_layout.addWidget(self.data_dir_edit)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_data_dir)
        data_dir_layout.addWidget(browse_button)
        
        layout.addRow("数据存储目录:", data_dir_layout)
        
        # 添加到标签页
        self.tab_widget.addTab(general_tab, "一般设置")
    
    def setup_device_tab(self):
        """设置设备设置标签页"""
        device_tab = QWidget()
        layout = QVBoxLayout(device_tab)
        
        # 设备选择
        device_group = QGroupBox("脑电设备")
        device_layout = QFormLayout(device_group)
        
        self.device_type_combo = QComboBox()
        self.device_type_combo.addItem("有线设备", "wired")
        self.device_type_combo.addItem("无线设备", "wireless")
        self.device_type_combo.addItem("模拟设备", "simulated")
        self.device_type_combo.currentIndexChanged.connect(self.on_device_type_changed)
        device_layout.addRow("设备类型:", self.device_type_combo)
        
        self.device_model_combo = QComboBox()
        # 实际设备型号将在选择设备类型后动态加载
        device_layout.addRow("设备型号:", self.device_model_combo)
        
        self.device_port_edit = QLineEdit()
        device_layout.addRow("设备端口:", self.device_port_edit)
        
        self.device_baudrate_combo = QComboBox()
        for rate in [9600, 19200, 38400, 57600, 115200]:
            self.device_baudrate_combo.addItem(str(rate), rate)
        device_layout.addRow("波特率:", self.device_baudrate_combo)
        
        layout.addWidget(device_group)
        
        # 采集参数
        acquisition_group = QGroupBox("采集参数")
        acquisition_layout = QFormLayout(acquisition_group)
        
        self.sample_rate_combo = QComboBox()
        for rate in [128, 256, 512, 1024]:
            self.sample_rate_combo.addItem(f"{rate} Hz", rate)
        acquisition_layout.addRow("采样率:", self.sample_rate_combo)
        
        self.channel_count_spin = QSpinBox()
        self.channel_count_spin.setRange(1, 32)
        acquisition_layout.addRow("通道数:", self.channel_count_spin)
        
        self.use_filters_check = QCheckBox("启用")
        acquisition_layout.addRow("使用滤波:", self.use_filters_check)
        
        layout.addWidget(acquisition_group)
        
        # 添加到标签页
        self.tab_widget.addTab(device_tab, "设备设置")
    
    def setup_user_tab(self):
        """设置用户信息标签页"""
        user_tab = QWidget()
        layout = QFormLayout(user_tab)
        
        # 用户名
        self.username_edit = QLineEdit()
        layout.addRow("姓名:", self.username_edit)
        
        # 年龄
        self.age_spin = QSpinBox()
        self.age_spin.setRange(1, 18)
        layout.addRow("年龄:", self.age_spin)
        
        # 性别
        self.gender_combo = QComboBox()
        self.gender_combo.addItem("男", "male")
        self.gender_combo.addItem("女", "female")
        self.gender_combo.addItem("其他", "other")
        layout.addRow("性别:", self.gender_combo)
        
        # 病史
        self.medical_history_check = QCheckBox("有注意力相关问题")
        layout.addRow("病史:", self.medical_history_check)
        
        # 备注
        self.note_edit = QLineEdit()
        layout.addRow("备注:", self.note_edit)
        
        # 添加到标签页
        self.tab_widget.addTab(user_tab, "用户信息")
    
    def setup_training_tab(self):
        """设置训练设置标签页"""
        training_tab = QWidget()
        layout = QFormLayout(training_tab)
        
        # 默认训练时长
        self.default_duration_spin = QSpinBox()
        self.default_duration_spin.setRange(1, 60)
        self.default_duration_spin.setValue(10)
        self.default_duration_spin.setSuffix(" 分钟")
        layout.addRow("默认训练时长:", self.default_duration_spin)
        
        # 默认难度
        self.default_difficulty_combo = QComboBox()
        for i in range(1, 6):
            self.default_difficulty_combo.addItem(f"{i}级", i)
        layout.addRow("默认难度:", self.default_difficulty_combo)
        
        # 自动调整难度
        self.auto_difficulty_check = QCheckBox("启用")
        layout.addRow("自动调整难度:", self.auto_difficulty_check)
        
        # 注意力阈值
        self.attention_threshold_spin = QSpinBox()
        self.attention_threshold_spin.setRange(0, 100)
        self.attention_threshold_spin.setValue(70)
        self.attention_threshold_spin.setSuffix(" /100")
        layout.addRow("注意力阈值:", self.attention_threshold_spin)
        
        # 训练计划
        self.training_plan_check = QCheckBox("启用个性化训练计划")
        layout.addRow("训练计划:", self.training_plan_check)
        
        # 反馈强度
        self.feedback_strength_slider = QDoubleSpinBox()
        self.feedback_strength_slider.setRange(0.1, 2.0)
        self.feedback_strength_slider.setValue(1.0)
        self.feedback_strength_slider.setSingleStep(0.1)
        layout.addRow("反馈强度:", self.feedback_strength_slider)
        
        # 添加到标签页
        self.tab_widget.addTab(training_tab, "训练设置")
    
    def load_settings(self):
        """从配置加载设置"""
        # 一般设置
        self.app_name_edit.setText(self.config.get("app_name", "儿童注意力康复训练系统"))
        
        log_level = self.config.get("log_level", "INFO")
        index = self.log_level_combo.findData(log_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        
        self.auto_save_check.setChecked(self.config.get("auto_save", True))
        self.data_dir_edit.setText(self.config.get("data_dir", "data"))
        
        # 设备设置
        device_config = self.config.get("device", {})
        
        device_type = device_config.get("type", "simulated")
        index = self.device_type_combo.findData(device_type)
        if index >= 0:
            self.device_type_combo.setCurrentIndex(index)
            self.update_device_models(device_type)
        
        device_model = device_config.get("model", "")
        index = self.device_model_combo.findData(device_model)
        if index >= 0:
            self.device_model_combo.setCurrentIndex(index)
        
        self.device_port_edit.setText(device_config.get("port", ""))
        
        baudrate = device_config.get("baudrate", 115200)
        index = self.device_baudrate_combo.findData(baudrate)
        if index >= 0:
            self.device_baudrate_combo.setCurrentIndex(index)
        
        acquisition_config = self.config.get("acquisition", {})
        
        sample_rate = acquisition_config.get("sample_rate", 256)
        index = self.sample_rate_combo.findData(sample_rate)
        if index >= 0:
            self.sample_rate_combo.setCurrentIndex(index)
        
        self.channel_count_spin.setValue(acquisition_config.get("channel_count", 8))
        self.use_filters_check.setChecked(acquisition_config.get("use_filters", True))
        
        # 用户信息
        user_config = self.config.get("user", {})
        
        self.username_edit.setText(user_config.get("name", ""))
        self.age_spin.setValue(user_config.get("age", 10))
        
        gender = user_config.get("gender", "male")
        index = self.gender_combo.findData(gender)
        if index >= 0:
            self.gender_combo.setCurrentIndex(index)
        
        self.medical_history_check.setChecked(user_config.get("has_medical_history", False))
        self.note_edit.setText(user_config.get("note", ""))
        
        # 训练设置
        training_config = self.config.get("training", {})
        
        self.default_duration_spin.setValue(training_config.get("default_duration", 10))
        
        difficulty = training_config.get("default_difficulty", 1)
        index = self.default_difficulty_combo.findData(difficulty)
        if index >= 0:
            self.default_difficulty_combo.setCurrentIndex(index)
        
        self.auto_difficulty_check.setChecked(training_config.get("auto_difficulty", True))
        self.attention_threshold_spin.setValue(training_config.get("attention_threshold", 70))
        self.training_plan_check.setChecked(training_config.get("use_training_plan", False))
        self.feedback_strength_slider.setValue(training_config.get("feedback_strength", 1.0))
    
    def save_settings(self):
        """保存设置到配置"""
        # 一般设置
        self.config["app_name"] = self.app_name_edit.text()
        self.config["log_level"] = self.log_level_combo.currentData()
        self.config["auto_save"] = self.auto_save_check.isChecked()
        self.config["data_dir"] = self.data_dir_edit.text()
        
        # 设备设置
        if "device" not in self.config:
            self.config["device"] = {}
        
        self.config["device"]["type"] = self.device_type_combo.currentData()
        self.config["device"]["model"] = self.device_model_combo.currentData() or ""
        self.config["device"]["port"] = self.device_port_edit.text()
        self.config["device"]["baudrate"] = self.device_baudrate_combo.currentData()
        
        if "acquisition" not in self.config:
            self.config["acquisition"] = {}
        
        self.config["acquisition"]["sample_rate"] = self.sample_rate_combo.currentData()
        self.config["acquisition"]["channel_count"] = self.channel_count_spin.value()
        self.config["acquisition"]["use_filters"] = self.use_filters_check.isChecked()
        
        # 用户信息
        if "user" not in self.config:
            self.config["user"] = {}
        
        self.config["user"]["name"] = self.username_edit.text()
        self.config["user"]["age"] = self.age_spin.value()
        self.config["user"]["gender"] = self.gender_combo.currentData()
        self.config["user"]["has_medical_history"] = self.medical_history_check.isChecked()
        self.config["user"]["note"] = self.note_edit.text()
        
        # 训练设置
        if "training" not in self.config:
            self.config["training"] = {}
        
        self.config["training"]["default_duration"] = self.default_duration_spin.value()
        self.config["training"]["default_difficulty"] = self.default_difficulty_combo.currentData()
        self.config["training"]["auto_difficulty"] = self.auto_difficulty_check.isChecked()
        self.config["training"]["attention_threshold"] = self.attention_threshold_spin.value()
        self.config["training"]["use_training_plan"] = self.training_plan_check.isChecked()
        self.config["training"]["feedback_strength"] = self.feedback_strength_slider.value()
        
        return self.config
    
    def accept(self):
        """确认对话框"""
        # 保存设置
        config = self.save_settings()
        
        # 发出设置变更信号
        self.settings_changed.emit(config)
        
        # 关闭对话框
        super().accept()
    
    def reset_settings(self):
        """重置设置"""
        reply = QMessageBox.question(
            self,
            "重置设置",
            "确定要重置所有设置吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config = self.original_config.copy()
            self.load_settings()
    
    def browse_data_dir(self):
        """浏览数据目录"""
        current_dir = self.data_dir_edit.text()
        if not os.path.isabs(current_dir):
            current_dir = os.path.join(os.getcwd(), current_dir)
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择数据存储目录",
            current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            # 尝试转换为相对路径
            try:
                cwd = os.getcwd()
                if directory.startswith(cwd):
                    directory = os.path.relpath(directory, cwd)
            except:
                pass
                
            self.data_dir_edit.setText(directory)
    
    def on_device_type_changed(self, index):
        """设备类型变更处理"""
        device_type = self.device_type_combo.currentData()
        self.update_device_models(device_type)
    
    def update_device_models(self, device_type):
        """更新设备型号列表"""
        self.device_model_combo.clear()
        
        if device_type == "wired":
            self.device_model_combo.addItem("OpenBCI Cyton", "cyton")
            self.device_model_combo.addItem("NeuroSky MindWave", "mindwave")
            self.device_model_combo.addItem("EmotivEPOC", "epoc")
        elif device_type == "wireless":
            self.device_model_combo.addItem("OpenBCI Ganglion", "ganglion")
            self.device_model_combo.addItem("Muse Headband", "muse")
            self.device_model_combo.addItem("EmotivInsight", "insight")
        else:  # simulated
            self.device_model_combo.addItem("随机波形", "random")
            self.device_model_combo.addItem("预录制数据", "recorded")
            self.device_model_combo.addItem("正弦波测试", "sine") 