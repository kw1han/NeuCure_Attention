#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
注意力仪表盘组件
显示实时注意力水平、脑电波和历史趋势
"""

import numpy as np
import time
from datetime import datetime
from collections import deque
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush

class AttentionGauge(QWidget):
    """注意力仪表盘控件"""
    
    def __init__(self, parent=None):
        """
        初始化注意力仪表
        
        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 设置最小尺寸
        self.setMinimumSize(200, 200)
        
        # 注意力等级和分数
        self.attention_level = "未知"
        self.attention_score = 0
        self.attention_normalized = 0
        
        # 配色
        self.colors = {
            "very_low": QColor(150, 0, 0),
            "low": QColor(230, 100, 0),
            "medium": QColor(230, 230, 0),
            "high": QColor(100, 230, 0),
            "very_high": QColor(0, 150, 0),
            "unknown": QColor(120, 120, 120)
        }
        
        # 设置背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)
    
    def update_attention(self, level, score, normalized_score):
        """
        更新注意力水平
        
        参数:
            level: 注意力水平
            score: 注意力分数(0-100)
            normalized_score: 归一化的注意力分数(0-100)
        """
        self.attention_level = level
        self.attention_score = score
        self.attention_normalized = normalized_score
        self.update()  # 触发重绘
    
    def paintEvent(self, event):
        """
        绘制控件
        
        参数:
            event: 绘制事件
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算中心和半径
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) - 10
        
        # 绘制外圆
        painter.setPen(QPen(Qt.black, 2))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # 绘制填充部分
        if self.attention_level in self.colors:
            color = self.colors[self.attention_level]
        else:
            color = self.colors["unknown"]
            
        # 计算填充角度（根据分数0-100映射到0-270度）
        start_angle = 135 * 16  # 开始于左下方（以1/100度为单位）
        span_angle = int(-(self.attention_normalized * 2.7) * 16)  # 顺时针方向（负值）
        
        # 填充扇形
        painter.setBrush(QBrush(color))
        painter.drawPie(center_x - radius, center_y - radius, radius * 2, radius * 2, start_angle, span_angle)
        
        # 绘制内圆（形成环形）
        inner_radius = radius * 0.7
        painter.setBrush(QBrush(Qt.white))
        painter.drawEllipse(center_x - inner_radius, center_y - inner_radius, inner_radius * 2, inner_radius * 2)
        
        # 绘制文字
        painter.setPen(Qt.black)
        
        # 分数
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        score_text = f"{int(self.attention_score)}"
        text_rect = painter.fontMetrics().boundingRect(score_text)
        painter.drawText(center_x - text_rect.width() / 2, center_y + text_rect.height() / 4, score_text)
        
        # 单位
        font = QFont("Arial", 10)
        painter.setFont(font)
        unit_text = "/100"
        text_rect = painter.fontMetrics().boundingRect(unit_text)
        painter.drawText(center_x + 25, center_y + 5, unit_text)
        
        # 等级
        level_text = self._translate_level(self.attention_level)
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        text_rect = painter.fontMetrics().boundingRect(level_text)
        painter.drawText(center_x - text_rect.width() / 2, center_y + 50, level_text)
    
    def _translate_level(self, level):
        """
        将英文等级翻译为中文
        
        参数:
            level: 英文等级
            
        返回:
            中文等级
        """
        translations = {
            "very_low": "极低",
            "low": "较低",
            "medium": "中等",
            "high": "较高",
            "very_high": "极高",
            "unknown": "未知"
        }
        
        return translations.get(level, "未知")

class EEGWaveform(pg.PlotWidget):
    """脑电波形图控件"""
    
    def __init__(self, parent=None):
        """
        初始化脑电波形图
        
        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 设置样式
        self.setBackground('w')
        self.setTitle("实时脑电波形")
        self.setLabel('left', "振幅")
        self.setLabel('bottom', "采样点")
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # 波形颜色
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        
        # 数据缓冲区
        self.buffer_size = 256 * 4  # 4秒数据
        self.data_buffers = None
        self.channel_count = 0
        
        # 波形线
        self.curves = []
    
    def setup_channels(self, channel_count, channel_names=None):
        """
        设置通道数
        
        参数:
            channel_count: 通道数量
            channel_names: 通道名称列表
        """
        self.channel_count = channel_count
        self.data_buffers = [np.zeros(self.buffer_size) for _ in range(channel_count)]
        
        # 创建曲线
        self.clear()
        self.curves = []
        
        # 添加图例
        self.addLegend()
        
        for i in range(channel_count):
            color = self.colors[i % len(self.colors)]
            name = channel_names[i] if channel_names and i < len(channel_names) else f"通道{i+1}"
            curve = self.plot(
                np.arange(self.buffer_size), 
                self.data_buffers[i], 
                pen=pg.mkPen(color=color, width=1), 
                name=name
            )
            self.curves.append(curve)
    
    def update_data(self, new_sample):
        """
        更新波形数据
        
        参数:
            new_sample: 新的EEG样本
        """
        if self.data_buffers is None or len(new_sample) != self.channel_count:
            # 如果通道数不匹配，重新设置
            self.setup_channels(len(new_sample))
        
        # 更新数据缓冲区（滚动）
        for i in range(self.channel_count):
            self.data_buffers[i] = np.roll(self.data_buffers[i], -1)
            self.data_buffers[i][-1] = new_sample[i]
            
            # 更新曲线数据
            self.curves[i].setData(np.arange(self.buffer_size), self.data_buffers[i])

class AttentionHistoryGraph(pg.PlotWidget):
    """注意力历史趋势图控件"""
    
    def __init__(self, parent=None):
        """
        初始化注意力历史趋势图
        
        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 设置样式
        self.setBackground('w')
        self.setTitle("注意力历史趋势")
        self.setLabel('left', "注意力得分")
        self.setLabel('bottom', "时间(秒)")
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Y轴范围
        self.setYRange(0, 100)
        
        # 数据缓冲区
        self.buffer_size = 300  # 5分钟数据(按1秒一个采样点)
        self.time_buffer = np.arange(self.buffer_size)
        self.score_buffer = np.zeros(self.buffer_size)
        
        # 创建曲线
        self.curve = self.plot(
            self.time_buffer, 
            self.score_buffer, 
            pen=pg.mkPen(color='b', width=2)
        )
        
        # 添加参考线
        self.addItem(pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen(color='r', style=Qt.DashLine, width=1)))
        self.addItem(pg.InfiniteLine(pos=40, angle=0, pen=pg.mkPen(color='y', style=Qt.DashLine, width=1)))
        self.addItem(pg.InfiniteLine(pos=60, angle=0, pen=pg.mkPen(color='y', style=Qt.DashLine, width=1)))
        self.addItem(pg.InfiniteLine(pos=80, angle=0, pen=pg.mkPen(color='g', style=Qt.DashLine, width=1)))
    
    def update_data(self, attention_score):
        """
        更新注意力分数数据
        
        参数:
            attention_score: 新的注意力分数
        """
        # 更新数据缓冲区（滚动）
        self.score_buffer = np.roll(self.score_buffer, -1)
        self.score_buffer[-1] = attention_score
        
        # 更新曲线数据
        self.curve.setData(self.time_buffer, self.score_buffer)

class AttentionStats(QWidget):
    """注意力统计信息控件"""
    
    def __init__(self, parent=None):
        """
        初始化注意力统计信息控件
        
        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 设置布局
        layout = QGridLayout(self)
        
        # 标签
        self.labels = {}
        
        # 创建统计信息标签
        stats = [
            ("current", "当前得分:", "0"),
            ("avg", "平均得分:", "0"),
            ("max", "最高得分:", "0"),
            ("min", "最低得分:", "0"),
            ("time_above", "高注意力时间:", "0:00"),
            ("time_below", "低注意力时间:", "0:00"),
            ("total_time", "总监测时间:", "0:00")
        ]
        
        for row, (key, title, value) in enumerate(stats):
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            value_label = QLabel(value)
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            font = QFont()
            font.setBold(True)
            value_label.setFont(font)
            
            layout.addWidget(title_label, row, 0)
            layout.addWidget(value_label, row, 1)
            
            self.labels[key] = value_label
    
    def update_stats(self, attention_history):
        """
        更新统计信息
        
        参数:
            attention_history: 注意力历史数据
        """
        if not attention_history:
            return
            
        # 提取分数数据
        scores = [item['score'] for item in attention_history]
        
        # 更新统计数据
        current = scores[-1]
        avg = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # 计算高低注意力时间
        high_attention_count = sum(1 for s in scores if s >= 60)
        low_attention_count = sum(1 for s in scores if s <= 40)
        
        # 假设每个数据点间隔为1秒
        high_attention_time = self._format_time(high_attention_count)
        low_attention_time = self._format_time(low_attention_count)
        total_time = self._format_time(len(scores))
        
        # 更新显示
        self.labels["current"].setText(f"{current:.1f}")
        self.labels["avg"].setText(f"{avg:.1f}")
        self.labels["max"].setText(f"{max_score:.1f}")
        self.labels["min"].setText(f"{min_score:.1f}")
        self.labels["time_above"].setText(high_attention_time)
        self.labels["time_below"].setText(low_attention_time)
        self.labels["total_time"].setText(total_time)
    
    def _format_time(self, seconds):
        """
        将秒数格式化为时间字符串
        
        参数:
            seconds: 秒数
            
        返回:
            格式化的时间字符串
        """
        minutes = seconds // 60
        seconds = seconds % 60
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

class AttentionDashboard(QWidget):
    """注意力仪表盘组件"""
    
    def __init__(self, parent=None):
        """
        初始化注意力仪表盘
        
        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 注意力历史数据
        self.attention_history = []
        
        # 设置布局
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主布局
        layout = QHBoxLayout(self)
        
        # 左侧布局（注意力仪表和统计信息）
        left_layout = QVBoxLayout()
        
        # 注意力仪表
        gauge_group = QGroupBox("注意力水平")
        gauge_layout = QVBoxLayout(gauge_group)
        self.attention_gauge = AttentionGauge()
        gauge_layout.addWidget(self.attention_gauge)
        left_layout.addWidget(gauge_group)
        
        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        self.attention_stats = AttentionStats()
        stats_layout.addWidget(self.attention_stats)
        left_layout.addWidget(stats_group)
        
        layout.addLayout(left_layout, 1)
        
        # 右侧布局（脑电波形和历史趋势）
        right_layout = QVBoxLayout()
        
        # 脑电波形图
        waveform_group = QGroupBox("脑电波形")
        waveform_layout = QVBoxLayout(waveform_group)
        self.eeg_waveform = EEGWaveform()
        waveform_layout.addWidget(self.eeg_waveform)
        right_layout.addWidget(waveform_group)
        
        # 注意力历史趋势
        history_group = QGroupBox("注意力趋势")
        history_layout = QVBoxLayout(history_group)
        self.attention_history_graph = AttentionHistoryGraph()
        history_layout.addWidget(self.attention_history_graph)
        right_layout.addWidget(history_group)
        
        layout.addLayout(right_layout, 3)
    
    @pyqtSlot(dict)
    def update_attention(self, attention_data):
        """
        更新注意力数据
        
        参数:
            attention_data: 注意力数据字典
        """
        # 更新注意力历史
        self.attention_history.append(attention_data)
        if len(self.attention_history) > 600:  # 保留最近10分钟的数据
            self.attention_history = self.attention_history[-600:]
        
        # 更新注意力仪表
        self.attention_gauge.update_attention(
            attention_data['level'],
            attention_data['score'],
            attention_data['normalized_score']
        )
        
        # 更新历史趋势图
        self.attention_history_graph.update_data(attention_data['score'])
        
        # 更新统计信息
        self.attention_stats.update_stats(self.attention_history)
    
    @pyqtSlot(np.ndarray)
    def update_eeg_data(self, eeg_data):
        """
        更新EEG数据
        
        参数:
            eeg_data: EEG数据数组
        """
        # 更新脑电波形图
        self.eeg_waveform.update_data(eeg_data) 