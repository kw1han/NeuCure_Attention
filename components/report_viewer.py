#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
报告查看器组件
提供训练报告查看、分析和导出功能
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QComboBox, QDateEdit, QGroupBox, QFormLayout,
    QTabWidget, QFileDialog, QMessageBox, QSplitter, QHeaderView
)
from PyQt5.QtCore import Qt, QDate, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QColor

import pyqtgraph as pg


class ReportViewer(QWidget):
    """报告查看器组件"""
    
    def __init__(self, config, parent=None):
        """
        初始化报告查看器
        
        参数:
            config: 配置信息
            parent: 父窗口
        """
        super().__init__(parent)
        
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.reports = []
        self.current_report = None
        
        self.setup_ui()
        
        # 加载报告数据
        self.load_reports()
    
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 过滤和控制区域
        control_layout = QHBoxLayout()
        
        # 报告过滤
        filter_group = QGroupBox("报告筛选")
        filter_layout = QFormLayout(filter_group)
        
        # 开始日期
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addDays(-30))  # 默认30天前
        self.start_date.dateChanged.connect(self.apply_filters)
        filter_layout.addRow("开始日期:", self.start_date)
        
        # 结束日期
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())  # 默认今天
        self.end_date.dateChanged.connect(self.apply_filters)
        filter_layout.addRow("结束日期:", self.end_date)
        
        # 训练类型
        self.game_type_combo = QComboBox()
        self.game_type_combo.addItem("全部", "all")
        self.game_type_combo.addItem("太空宝贝", "space_baby")
        self.game_type_combo.addItem("魔法森林大冒险", "magic_forest")
        self.game_type_combo.addItem("色彩拼图奇遇", "color_puzzle")
        self.game_type_combo.currentIndexChanged.connect(self.apply_filters)
        filter_layout.addRow("训练类型:", self.game_type_combo)
        
        control_layout.addWidget(filter_group)
        
        # 操作按钮
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        
        self.refresh_button = QPushButton("刷新报告")
        self.refresh_button.clicked.connect(self.load_reports)
        button_layout.addWidget(self.refresh_button)
        
        self.export_button = QPushButton("导出报告")
        self.export_button.clicked.connect(self.export_report)
        button_layout.addWidget(self.export_button)
        
        self.delete_button = QPushButton("删除报告")
        self.delete_button.clicked.connect(self.delete_report)
        button_layout.addWidget(self.delete_button)
        
        control_layout.addLayout(button_layout)
        
        layout.addLayout(control_layout)
        
        # 分割器
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter, 1)
        
        # 报告列表
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.report_table = QTableWidget()
        self.report_table.setColumnCount(7)
        self.report_table.setHorizontalHeaderLabels([
            "日期", "时间", "训练类型", "持续时间", "平均注意力", "最高注意力", "得分"
        ])
        
        # 设置表格样式
        self.report_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.report_table.setSelectionMode(QTableWidget.SingleSelection)
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.report_table.setAlternatingRowColors(True)
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.report_table.verticalHeader().setVisible(False)
        
        self.report_table.selectionModel().selectionChanged.connect(self.on_report_selected)
        
        table_layout.addWidget(self.report_table)
        
        splitter.addWidget(table_widget)
        
        # 报告详情
        self.detail_tabs = QTabWidget()
        
        # 概述标签页
        self.summary_tab = QWidget()
        self.setup_summary_tab()
        self.detail_tabs.addTab(self.summary_tab, "概述")
        
        # 图表标签页
        self.chart_tab = QWidget()
        self.setup_chart_tab()
        self.detail_tabs.addTab(self.chart_tab, "详细数据")
        
        # 进度标签页
        self.progress_tab = QWidget()
        self.setup_progress_tab()
        self.detail_tabs.addTab(self.progress_tab, "训练进度")
        
        splitter.addWidget(self.detail_tabs)
        
        # 设置分割器比例
        splitter.setSizes([200, 400])
    
    def setup_summary_tab(self):
        """设置概述标签页"""
        layout = QVBoxLayout(self.summary_tab)
        
        self.summary_form = QFormLayout()
        
        # 训练日期时间
        self.summary_datetime = QLabel("-")
        self.summary_form.addRow("训练日期时间:", self.summary_datetime)
        
        # 训练类型
        self.summary_game_type = QLabel("-")
        self.summary_form.addRow("训练类型:", self.summary_game_type)
        
        # 训练时长
        self.summary_duration = QLabel("-")
        self.summary_form.addRow("训练时长:", self.summary_duration)
        
        # 平均注意力
        self.summary_avg_attention = QLabel("-")
        self.summary_form.addRow("平均注意力:", self.summary_avg_attention)
        
        # 最高注意力
        self.summary_max_attention = QLabel("-")
        self.summary_form.addRow("最高注意力:", self.summary_max_attention)
        
        # 持续注意力时间
        self.summary_sustained = QLabel("-")
        self.summary_form.addRow("持续专注时间:", self.summary_sustained)
        
        # 最高难度
        self.summary_max_difficulty = QLabel("-")
        self.summary_form.addRow("达到最高难度:", self.summary_max_difficulty)
        
        # 总得分
        self.summary_score = QLabel("-")
        self.summary_form.addRow("总得分:", self.summary_score)
        
        layout.addLayout(self.summary_form)
        layout.addStretch()
    
    def setup_chart_tab(self):
        """设置图表标签页"""
        layout = QVBoxLayout(self.chart_tab)
        
        # 注意力变化图
        attention_group = QGroupBox("注意力变化")
        attention_layout = QVBoxLayout(attention_group)
        
        self.attention_plot = pg.PlotWidget()
        self.attention_plot.setBackground('w')
        self.attention_plot.setLabel('left', "注意力")
        self.attention_plot.setLabel('bottom', "时间(秒)")
        self.attention_plot.showGrid(x=True, y=True, alpha=0.3)
        
        attention_layout.addWidget(self.attention_plot)
        
        layout.addWidget(attention_group)
        
        # 训练难度图
        difficulty_group = QGroupBox("训练难度")
        difficulty_layout = QVBoxLayout(difficulty_group)
        
        self.difficulty_plot = pg.PlotWidget()
        self.difficulty_plot.setBackground('w')
        self.difficulty_plot.setLabel('left', "难度")
        self.difficulty_plot.setLabel('bottom', "时间(秒)")
        self.difficulty_plot.showGrid(x=True, y=True, alpha=0.3)
        
        difficulty_layout.addWidget(self.difficulty_plot)
        
        layout.addWidget(difficulty_group)
    
    def setup_progress_tab(self):
        """设置进度标签页"""
        layout = QVBoxLayout(self.progress_tab)
        
        # 平均注意力趋势
        avg_attention_group = QGroupBox("平均注意力趋势")
        avg_attention_layout = QVBoxLayout(avg_attention_group)
        
        self.avg_attention_plot = pg.PlotWidget()
        self.avg_attention_plot.setBackground('w')
        self.avg_attention_plot.setLabel('left', "平均注意力")
        self.avg_attention_plot.setLabel('bottom', "训练日期")
        self.avg_attention_plot.showGrid(x=True, y=True, alpha=0.3)
        
        avg_attention_layout.addWidget(self.avg_attention_plot)
        
        layout.addWidget(avg_attention_group)
        
        # 持续注意力趋势
        sustained_group = QGroupBox("持续注意力趋势")
        sustained_layout = QVBoxLayout(sustained_group)
        
        self.sustained_plot = pg.PlotWidget()
        self.sustained_plot.setBackground('w')
        self.sustained_plot.setLabel('left', "持续注意力时间(秒)")
        self.sustained_plot.setLabel('bottom', "训练日期")
        self.sustained_plot.showGrid(x=True, y=True, alpha=0.3)
        
        sustained_layout.addWidget(self.sustained_plot)
        
        layout.addWidget(sustained_group)
    
    def load_reports(self):
        """加载训练报告"""
        self.reports = []
        
        reports_dir = os.path.join(self.data_dir, "reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir, exist_ok=True)
            return
        
        # 读取所有报告文件
        for filename in os.listdir(reports_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(reports_dir, filename), 'r', encoding='utf-8') as f:
                        report = json.load(f)
                        self.reports.append(report)
                except Exception as e:
                    print(f"Error loading report {filename}: {e}")
        
        # 按时间戳排序，最新的在前
        self.reports.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # 更新表格
        self.update_report_table()
        
        # 更新训练进度图表
        self.update_progress_charts()
    
    def update_report_table(self):
        """更新报告表格"""
        self.report_table.setRowCount(0)
        
        filtered_reports = self.filter_reports()
        
        for i, report in enumerate(filtered_reports):
            self.report_table.insertRow(i)
            
            # 转换时间戳为日期时间
            dt = datetime.datetime.fromtimestamp(report.get('timestamp', 0))
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M:%S')
            
            # 训练类型
            game_type = report.get('game_type', '')
            game_type_display = self.translate_game_type(game_type)
            
            # 训练时长
            duration = report.get('duration', 0)
            duration_str = self.format_duration(duration)
            
            # 注意力数据
            avg_attention = report.get('avg_attention', 0)
            max_attention = report.get('max_attention', 0)
            
            # 得分
            score = report.get('score', 0)
            
            # 设置表格单元格
            self.report_table.setItem(i, 0, QTableWidgetItem(date_str))
            self.report_table.setItem(i, 1, QTableWidgetItem(time_str))
            self.report_table.setItem(i, 2, QTableWidgetItem(game_type_display))
            self.report_table.setItem(i, 3, QTableWidgetItem(duration_str))
            self.report_table.setItem(i, 4, QTableWidgetItem(f"{avg_attention:.1f}"))
            self.report_table.setItem(i, 5, QTableWidgetItem(f"{max_attention:.1f}"))
            self.report_table.setItem(i, 6, QTableWidgetItem(str(score)))
            
            # 设置行背景色，根据平均注意力
            if avg_attention >= 80:
                for col in range(7):
                    self.report_table.item(i, col).setBackground(QColor(200, 255, 200))  # 浅绿色
            elif avg_attention >= 60:
                for col in range(7):
                    self.report_table.item(i, col).setBackground(QColor(255, 255, 200))  # 浅黄色
        
        # 如果有报告，默认选中第一行
        if self.report_table.rowCount() > 0:
            self.report_table.selectRow(0)
    
    def filter_reports(self):
        """筛选报告"""
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        end_date = datetime.datetime.combine(end_date, datetime.time(23, 59, 59))
        
        game_type = self.game_type_combo.currentData()
        
        # 转换日期为时间戳
        start_ts = datetime.datetime.combine(start_date, datetime.time.min).timestamp()
        end_ts = end_date.timestamp()
        
        filtered = []
        for report in self.reports:
            ts = report.get('timestamp', 0)
            if start_ts <= ts <= end_ts:
                if game_type == "all" or report.get('game_type', '') == game_type:
                    filtered.append(report)
        
        return filtered
    
    def apply_filters(self):
        """应用筛选"""
        self.update_report_table()
    
    def on_report_selected(self):
        """处理报告选中事件"""
        selected_rows = self.report_table.selectionModel().selectedRows()
        if not selected_rows:
            self.clear_detail_view()
            return
        
        row = selected_rows[0].row()
        filtered_reports = self.filter_reports()
        
        if row < 0 or row >= len(filtered_reports):
            self.clear_detail_view()
            return
        
        self.current_report = filtered_reports[row]
        self.update_detail_view()
    
    def clear_detail_view(self):
        """清空详情视图"""
        # 概述
        self.summary_datetime.setText("-")
        self.summary_game_type.setText("-")
        self.summary_duration.setText("-")
        self.summary_avg_attention.setText("-")
        self.summary_max_attention.setText("-")
        self.summary_sustained.setText("-")
        self.summary_max_difficulty.setText("-")
        self.summary_score.setText("-")
        
        # 图表
        self.attention_plot.clear()
        self.difficulty_plot.clear()
    
    def update_detail_view(self):
        """更新详情视图"""
        if not self.current_report:
            return
        
        # 更新概述
        dt = datetime.datetime.fromtimestamp(self.current_report.get('timestamp', 0))
        self.summary_datetime.setText(dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        game_type = self.current_report.get('game_type', '')
        self.summary_game_type.setText(self.translate_game_type(game_type))
        
        duration = self.current_report.get('duration', 0)
        self.summary_duration.setText(self.format_duration(duration))
        
        avg_attention = self.current_report.get('avg_attention', 0)
        self.summary_avg_attention.setText(f"{avg_attention:.1f} / 100")
        
        max_attention = self.current_report.get('max_attention', 0)
        self.summary_max_attention.setText(f"{max_attention:.1f} / 100")
        
        sustained = self.current_report.get('sustained_periods', 0)
        self.summary_sustained.setText(f"{sustained} 秒")
        
        max_difficulty = self.current_report.get('max_difficulty', 1)
        self.summary_max_difficulty.setText(f"{max_difficulty} 级")
        
        score = self.current_report.get('score', 0)
        self.summary_score.setText(str(score))
        
        # 更新图表
        self.update_detail_charts()
    
    def update_detail_charts(self):
        """更新详情图表"""
        if not self.current_report:
            return
        
        # 注意力变化图
        self.attention_plot.clear()
        
        attention_history = self.current_report.get('attention_history', [])
        if attention_history:
            x = list(range(len(attention_history)))
            y = attention_history
            
            pen = pg.mkPen(color=(50, 100, 200), width=2)
            self.attention_plot.plot(x, y, pen=pen)
            
            # 添加水平阈值线 (70分为高注意力阈值)
            threshold_line = pg.InfiniteLine(
                pos=70, angle=0, 
                pen=pg.mkPen(color=(200, 50, 50), width=1, style=Qt.DashLine),
                label="高注意力阈值", labelOpts={'position': 0.9, 'color': (200, 50, 50)}
            )
            self.attention_plot.addItem(threshold_line)
        
        # 难度变化图
        self.difficulty_plot.clear()
        
        difficulty_history = self.current_report.get('difficulty_history', [])
        if difficulty_history:
            x = list(range(len(difficulty_history)))
            y = difficulty_history
            
            pen = pg.mkPen(color=(100, 200, 50), width=2)
            self.difficulty_plot.plot(x, y, pen=pen, stepMode=True)
    
    def update_progress_charts(self):
        """更新进度图表"""
        if not self.reports:
            return
        
        # 按日期分组并计算平均值
        dates = []
        avg_attentions = []
        sustained_times = []
        
        # 将报告按日期分组
        reports_by_date = {}
        for report in self.reports:
            ts = report.get('timestamp', 0)
            dt = datetime.datetime.fromtimestamp(ts)
            date_str = dt.strftime('%Y-%m-%d')
            
            if date_str not in reports_by_date:
                reports_by_date[date_str] = []
            
            reports_by_date[date_str].append(report)
        
        # 计算每天的平均值
        for date_str, daily_reports in sorted(reports_by_date.items()):
            dates.append(date_str)
            
            # 平均注意力
            avg_attention = np.mean([r.get('avg_attention', 0) for r in daily_reports])
            avg_attentions.append(avg_attention)
            
            # 持续注意力
            sustained_time = np.mean([r.get('sustained_periods', 0) for r in daily_reports])
            sustained_times.append(sustained_time)
        
        # 更新平均注意力趋势图
        self.avg_attention_plot.clear()
        if dates and avg_attentions:
            x = list(range(len(dates)))
            y = avg_attentions
            
            pen = pg.mkPen(color=(50, 100, 200), width=2)
            self.avg_attention_plot.plot(x, y, pen=pen, symbol='o', symbolSize=8, symbolBrush=(50, 100, 200))
            
            # 添加日期标签
            axis = self.avg_attention_plot.getAxis('bottom')
            ticks = [(i, date) for i, date in enumerate(dates)]
            axis.setTicks([ticks])
            
            # 添加趋势线
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                self.avg_attention_plot.plot(x, p(x), pen=pg.mkPen(color=(200, 50, 50), width=2, style=Qt.DashLine))
        
        # 更新持续注意力趋势图
        self.sustained_plot.clear()
        if dates and sustained_times:
            x = list(range(len(dates)))
            y = sustained_times
            
            pen = pg.mkPen(color=(100, 200, 50), width=2)
            self.sustained_plot.plot(x, y, pen=pen, symbol='o', symbolSize=8, symbolBrush=(100, 200, 50))
            
            # 添加日期标签
            axis = self.sustained_plot.getAxis('bottom')
            ticks = [(i, date) for i, date in enumerate(dates)]
            axis.setTicks([ticks])
            
            # 添加趋势线
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                self.sustained_plot.plot(x, p(x), pen=pg.mkPen(color=(200, 50, 50), width=2, style=Qt.DashLine))
    
    def export_report(self):
        """导出报告"""
        if not self.reports:
            QMessageBox.information(self, "提示", "没有可导出的报告数据")
            return
        
        # 选择保存文件
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出报告",
            os.path.join(os.path.expanduser("~"), "训练报告.csv"),
            "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
        )
        
        if not filename:
            return
        
        try:
            # 准备数据
            data = []
            for report in self.reports:
                dt = datetime.datetime.fromtimestamp(report.get('timestamp', 0))
                
                data.append({
                    "日期": dt.strftime('%Y-%m-%d'),
                    "时间": dt.strftime('%H:%M:%S'),
                    "训练类型": self.translate_game_type(report.get('game_type', '')),
                    "训练时长(秒)": report.get('duration', 0),
                    "平均注意力": report.get('avg_attention', 0),
                    "最高注意力": report.get('max_attention', 0),
                    "持续注意力时间(秒)": report.get('sustained_periods', 0),
                    "最高难度": report.get('max_difficulty', 1),
                    "得分": report.get('score', 0)
                })
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 保存文件
            if filename.endswith('.csv'):
                df.to_csv(filename, index=False, encoding='utf-8-sig')
            else:
                df.to_excel(filename, index=False)
            
            QMessageBox.information(self, "成功", "报告导出成功")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出报告失败：{str(e)}")
    
    def delete_report(self):
        """删除报告"""
        selected_rows = self.report_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择要删除的报告")
            return
        
        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除所选报告吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        row = selected_rows[0].row()
        filtered_reports = self.filter_reports()
        
        if row < 0 or row >= len(filtered_reports):
            return
        
        report = filtered_reports[row]
        
        # 删除报告文件
        try:
            reports_dir = os.path.join(self.data_dir, "reports")
            
            # 根据时间戳查找文件
            ts = report.get('timestamp', 0)
            filename = None
            
            for f in os.listdir(reports_dir):
                if f.endswith(".json"):
                    file_path = os.path.join(reports_dir, f)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if data.get('timestamp') == ts:
                            filename = f
                            break
            
            if filename:
                os.remove(os.path.join(reports_dir, filename))
                
                # 从列表中删除
                for i, r in enumerate(self.reports):
                    if r.get('timestamp') == ts:
                        del self.reports[i]
                        break
                
                # 更新界面
                self.update_report_table()
                self.update_progress_charts()
                
                QMessageBox.information(self, "成功", "报告已删除")
            else:
                QMessageBox.warning(self, "警告", "未找到对应的报告文件")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除报告失败：{str(e)}")
    
    def translate_game_type(self, game_type):
        """转换游戏类型为中文名称"""
        translations = {
            "space_baby": "太空宝贝",
            "magic_forest": "魔法森林大冒险",
            "color_puzzle": "色彩拼图奇遇"
        }
        
        return translations.get(game_type, game_type)
    
    def format_duration(self, seconds):
        """格式化时长"""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            return f"{h}时{m}分{s}秒"
        elif m > 0:
            return f"{m}分{s}秒"
        else:
            return f"{s}秒"