#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练游戏组件
提供基于注意力的互动训练游戏界面
"""

import random
import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFrame, QStackedWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QBrush, QColor, QPen, QFont, QPainter

class TrainingGameWidget(QWidget):
    """训练游戏组件"""

    # 定义信号
    game_completed = pyqtSignal(dict)  # 游戏完成信号，传递游戏结果
    attention_required = pyqtSignal(int)  # 需要注意力数据信号，传递需要的注意力级别

    def __init__(self, config, parent=None):
        """
        初始化训练游戏组件
        
        参数:
            config: 配置信息
            parent: 父窗口
        """
        super().__init__(parent)
        
        self.config = config
        self.current_game = None
        self.game_running = False
        self.game_duration = 600  # 默认游戏时长10分钟（秒）
        self.remaining_time = 0
        self.difficulty = 1
        self.auto_difficulty = True
        self.attention_history = []
        self.score = 0
        
        # 游戏实例
        self.games = {
            "space_baby": SpaceBabyGame(self),
            "magic_forest": MagicForestGame(self),
            "color_puzzle": ColorPuzzleGame(self)
        }
        
        self.setup_ui()
        
        # 游戏定时器
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game_time)
        
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 游戏状态区域
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        
        # 得分标签
        self.score_label = QLabel("得分: 0")
        self.score_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.score_label)
        
        # 当前注意力标签
        self.attention_label = QLabel("注意力: --")
        self.attention_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.attention_label)
        
        # 难度标签
        self.difficulty_label = QLabel("难度: 1")
        status_layout.addWidget(self.difficulty_label)
        
        # 剩余时间标签和进度条
        status_layout.addWidget(QLabel("剩余时间:"))
        
        self.time_progress = QProgressBar()
        self.time_progress.setRange(0, 100)
        self.time_progress.setValue(100)
        self.time_progress.setTextVisible(True)
        self.time_progress.setFormat("%v%")
        status_layout.addWidget(self.time_progress, 1)
        
        self.time_label = QLabel("10:00")
        self.time_label.setMinimumWidth(50)
        status_layout.addWidget(self.time_label)
        
        # 暂停/继续按钮
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        status_layout.addWidget(self.pause_button)
        
        layout.addWidget(status_frame)
        
        # 游戏区域 - 使用堆叠部件以支持不同的游戏
        self.game_stack = QStackedWidget()
        
        # 游戏开始提示界面
        self.start_widget = QWidget()
        start_layout = QVBoxLayout(self.start_widget)
        start_layout.addStretch()
        
        start_label = QLabel("请点击\"开始训练\"按钮开始游戏训练")
        start_label.setAlignment(Qt.AlignCenter)
        start_label.setFont(QFont("Arial", 18, QFont.Bold))
        start_layout.addWidget(start_label)
        
        instruction_label = QLabel("通过集中注意力来控制游戏角色，提高注意力水平以获得更高分数。")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setWordWrap(True)
        instruction_label.setFont(QFont("Arial", 12))
        start_layout.addWidget(instruction_label)
        
        start_layout.addStretch()
        
        self.game_stack.addWidget(self.start_widget)
        
        # 添加各个游戏界面到堆叠部件
        for game in self.games.values():
            self.game_stack.addWidget(game)
        
        layout.addWidget(self.game_stack, 1)
        
    def start_game(self, game_key, duration_minutes, difficulty, auto_difficulty):
        """
        开始游戏
        
        参数:
            game_key: 游戏标识
            duration_minutes: 游戏时长（分钟）
            difficulty: 难度级别
            auto_difficulty: 是否自动调节难度
        """
        if game_key not in self.games:
            return False
        
        self.game_running = True
        self.current_game = game_key
        self.game_duration = duration_minutes * 60
        self.remaining_time = self.game_duration
        self.difficulty = difficulty
        self.auto_difficulty = auto_difficulty
        self.score = 0
        self.attention_history = []
        
        # 更新UI
        self.score_label.setText(f"得分: {self.score}")
        self.difficulty_label.setText(f"难度: {self.difficulty}")
        self.update_time_display()
        
        # 切换到对应游戏界面
        self.game_stack.setCurrentWidget(self.games[game_key])
        
        # 启动游戏
        self.games[game_key].start_game(difficulty)
        
        # 启动定时器
        self.game_timer.start(1000)  # 每秒更新一次
        
        # 启用暂停按钮
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停")
        
        return True
    
    def end_game(self):
        """结束游戏"""
        if not self.game_running:
            return
        
        self.game_running = False
        self.game_timer.stop()
        
        # 停止当前游戏
        if self.current_game:
            self.games[self.current_game].stop_game()
        
        # 切换回开始界面
        self.game_stack.setCurrentWidget(self.start_widget)
        
        # 禁用暂停按钮
        self.pause_button.setEnabled(False)
        
        # 计算游戏结果
        avg_attention = np.mean(self.attention_history) if self.attention_history else 0
        max_attention = np.max(self.attention_history) if self.attention_history else 0
        sustained_periods = self._calculate_sustained_attention()
        
        game_results = {
            "game_type": self.current_game,
            "duration": self.game_duration - self.remaining_time,
            "score": self.score,
            "max_difficulty": self.difficulty,
            "avg_attention": avg_attention,
            "max_attention": max_attention,
            "sustained_periods": sustained_periods,
            "timestamp": time.time()
        }
        
        # 发出游戏完成信号
        self.game_completed.emit(game_results)
    
    def toggle_pause(self):
        """暂停/继续游戏"""
        if not self.game_running:
            return
        
        if self.game_timer.isActive():
            # 暂停游戏
            self.game_timer.stop()
            if self.current_game:
                self.games[self.current_game].pause_game()
            self.pause_button.setText("继续")
        else:
            # 继续游戏
            self.game_timer.start(1000)
            if self.current_game:
                self.games[self.current_game].resume_game()
            self.pause_button.setText("暂停")
    
    @pyqtSlot()
    def update_game_time(self):
        """更新游戏时间"""
        if self.remaining_time > 0:
            self.remaining_time -= 1
            self.update_time_display()
        else:
            # 游戏时间结束
            self.end_game()
    
    def update_time_display(self):
        """更新时间显示"""
        minutes = self.remaining_time // 60
        seconds = self.remaining_time % 60
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
        
        # 更新进度条
        progress = (self.remaining_time / self.game_duration) * 100
        self.time_progress.setValue(int(progress))
    
    @pyqtSlot(dict)
    def update_attention(self, attention_data):
        """
        更新注意力数据
        
        参数:
            attention_data: 注意力数据字典
        """
        if not self.game_running:
            return
        
        # 更新注意力显示
        score = attention_data.get("score", 0)
        level = attention_data.get("level", "unknown")
        
        self.attention_label.setText(f"注意力: {int(score)}")
        
        # 记录注意力历史
        self.attention_history.append(score)
        
        # 调整游戏难度（如果启用自动难度）
        if self.auto_difficulty and len(self.attention_history) >= 5:
            # 每5秒根据平均注意力调整一次难度
            recent_attention = np.mean(self.attention_history[-5:])
            
            # 根据注意力水平自动调整难度
            new_difficulty = self.difficulty
            if recent_attention > 80:
                new_difficulty = min(5, self.difficulty + 1)
            elif recent_attention < 30:
                new_difficulty = max(1, self.difficulty - 1)
            
            if new_difficulty != self.difficulty:
                self.difficulty = new_difficulty
                self.difficulty_label.setText(f"难度: {self.difficulty}")
                
                # 更新游戏难度
                if self.current_game:
                    self.games[self.current_game].set_difficulty(self.difficulty)
        
        # 将注意力数据传递给当前游戏
        if self.current_game:
            self.games[self.current_game].process_attention(score, level)
    
    def add_score(self, points):
        """
        增加分数
        
        参数:
            points: 增加的分数
        """
        self.score += points
        self.score_label.setText(f"得分: {self.score}")
    
    def _calculate_sustained_attention(self):
        """
        计算持续注意力的时间段
        
        返回:
            持续注意力高于阈值的时间段（秒）
        """
        if not self.attention_history:
            return 0
        
        # 定义高注意力阈值
        threshold = 70
        
        # 计算持续高注意力的时长
        sustained_seconds = 0
        current_streak = 0
        
        for attention in self.attention_history:
            if attention >= threshold:
                current_streak += 1
                # 只有连续3秒以上才计入持续时间
                if current_streak >= 3:
                    sustained_seconds += 1
            else:
                current_streak = 0
        
        return sustained_seconds


class BaseGame(QWidget):
    """游戏基类"""
    
    def __init__(self, parent):
        """
        初始化游戏
        
        参数:
            parent: 父窗口
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.game_paused = False
        self.difficulty = 1
        
    def start_game(self, difficulty):
        """
        开始游戏
        
        参数:
            difficulty: 难度级别
        """
        self.difficulty = difficulty
        self.game_paused = False
        
    def stop_game(self):
        """停止游戏"""
        pass
        
    def pause_game(self):
        """暂停游戏"""
        self.game_paused = True
        
    def resume_game(self):
        """继续游戏"""
        self.game_paused = False
        
    def set_difficulty(self, difficulty):
        """
        设置难度
        
        参数:
            difficulty: 难度级别
        """
        self.difficulty = difficulty
        
    def process_attention(self, score, level):
        """
        处理注意力数据
        
        参数:
            score: 注意力分数
            level: 注意力等级
        """
        pass


class SpaceBabyGame(BaseGame):
    """太空宝贝游戏"""
    
    def __init__(self, parent):
        """初始化游戏"""
        super().__init__(parent)
        
        self.setStyleSheet("background-color: #000033;")
        
        # 游戏参数
        self.rocket_position = 0.5  # 相对位置 (0.0-1.0)
        self.rocket_target = 0.5  # 目标位置
        self.rocket_speed = 0.01  # 移动速度
        self.stars = []  # 星星列表
        self.meteors = []  # 陨石列表
        self.collected_stars = 0
        
        # 游戏区视图
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.view)
        
        # 加载图像
        self.rocket_pixmap = QPixmap("assets/rocket.png")
        if self.rocket_pixmap.isNull():
            # 如果图像加载失败，创建一个简单的替代图形
            self.rocket_pixmap = QPixmap(60, 80)
            self.rocket_pixmap.fill(Qt.transparent)
            painter = QPainter(self.rocket_pixmap)
            painter.setPen(QPen(Qt.red, 2))
            painter.setBrush(QBrush(Qt.yellow))
            painter.drawRect(10, 10, 40, 60)
            painter.end()
        
        # 游戏元素
        self.rocket = None
        
        # 游戏定时器
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        
        # 生成物体定时器
        self.spawn_timer = QTimer(self)
        self.spawn_timer.timeout.connect(self.spawn_objects)
        
    def start_game(self, difficulty):
        """开始游戏"""
        super().start_game(difficulty)
        
        # 重置游戏状态
        self.rocket_position = 0.5
        self.rocket_target = 0.5
        self.stars = []
        self.meteors = []
        self.collected_stars = 0
        
        # 设置场景
        self.scene.clear()
        scene_rect = QRectF(0, 0, self.width(), self.height())
        self.scene.setSceneRect(scene_rect)
        
        # 创建火箭
        self.rocket = QGraphicsPixmapItem(self.rocket_pixmap.scaled(60, 80, Qt.KeepAspectRatio))
        self.rocket.setPos((self.width() - 60) / 2, self.height() - 100)
        self.scene.addItem(self.rocket)
        
        # 启动定时器
        self.game_timer.start(33)  # 约30 FPS
        self.spawn_timer.start(1000 // self.difficulty)  # 根据难度调整生成频率
        
    def stop_game(self):
        """停止游戏"""
        self.game_timer.stop()
        self.spawn_timer.stop()
        
    def pause_game(self):
        """暂停游戏"""
        super().pause_game()
        self.game_timer.stop()
        self.spawn_timer.stop()
        
    def resume_game(self):
        """继续游戏"""
        super().resume_game()
        self.game_timer.start(33)
        self.spawn_timer.start(1000 // self.difficulty)
        
    def set_difficulty(self, difficulty):
        """设置难度"""
        super().set_difficulty(difficulty)
        if not self.game_paused:
            self.spawn_timer.start(1000 // self.difficulty)
        
    def process_attention(self, score, level):
        """处理注意力数据"""
        # 根据注意力水平控制火箭目标位置
        # 注意力越高，火箭越能向上移动
        attention_normalized = min(1.0, max(0.0, score / 100.0))
        
        # 目标位置根据注意力水平进行映射
        # 注意力0对应底部，100对应顶部，但保留一些边界
        self.rocket_target = 0.9 - 0.8 * attention_normalized  # 相对位置 (0.1-0.9)
        
    def update_game(self):
        """更新游戏状态"""
        if self.game_paused:
            return
            
        # 更新火箭位置
        if abs(self.rocket_position - self.rocket_target) > 0.001:
            # 平滑移动到目标位置
            if self.rocket_position < self.rocket_target:
                self.rocket_position += min(self.rocket_speed, self.rocket_target - self.rocket_position)
            else:
                self.rocket_position -= min(self.rocket_speed, self.rocket_position - self.rocket_target)
        
        # 计算火箭实际Y坐标（相对于场景高度）
        rocket_y = self.height() * self.rocket_position
        if self.rocket:
            self.rocket.setPos((self.width() - 60) / 2, rocket_y)
        
        # 更新星星和陨石
        self.update_objects()
        
    def spawn_objects(self):
        """生成游戏物体"""
        if self.game_paused:
            return
            
        # 每次生成 1-3 个星星
        for _ in range(random.randint(1, 3)):
            star = {
                'x': random.randint(20, self.width() - 20),
                'y': -20,
                'size': random.randint(15, 25),
                'speed': 2 + random.random() * 2 * self.difficulty,
                'item': None
            }
            
            # 创建图形项
            ellipse = QGraphicsEllipseItem(0, 0, star['size'], star['size'])
            ellipse.setBrush(QBrush(QColor(255, 255, 0)))
            ellipse.setPen(QPen(Qt.white, 1))
            ellipse.setPos(star['x'], star['y'])
            self.scene.addItem(ellipse)
            
            star['item'] = ellipse
            self.stars.append(star)
        
        # 根据难度生成陨石
        if random.random() < 0.2 * self.difficulty:
            meteor = {
                'x': random.randint(0, self.width()),
                'y': -40,
                'size': random.randint(25, 40),
                'speed': 3 + random.random() * 3 * self.difficulty,
                'item': None
            }
            
            # 创建图形项
            ellipse = QGraphicsEllipseItem(0, 0, meteor['size'], meteor['size'])
            ellipse.setBrush(QBrush(QColor(150, 100, 100)))
            ellipse.setPen(QPen(Qt.darkRed, 2))
            ellipse.setPos(meteor['x'], meteor['y'])
            self.scene.addItem(ellipse)
            
            meteor['item'] = ellipse
            self.meteors.append(meteor)
            
    def update_objects(self):
        """更新游戏物体"""
        rocket_rect = self.rocket.sceneBoundingRect() if self.rocket else None
        
        # 更新星星
        stars_to_remove = []
        for star in self.stars:
            star['y'] += star['speed']
            star['item'].setPos(star['x'], star['y'])
            
            # 检查是否超出屏幕
            if star['y'] > self.height():
                stars_to_remove.append(star)
            
            # 检查是否被火箭收集
            elif rocket_rect and star['item'].sceneBoundingRect().intersects(rocket_rect):
                stars_to_remove.append(star)
                self.collected_stars += 1
                
                # 增加得分
                self.parent_widget.add_score(10 * self.difficulty)
        
        # 删除超出屏幕或被收集的星星
        for star in stars_to_remove:
            self.scene.removeItem(star['item'])
            self.stars.remove(star)
        
        # 更新陨石
        meteors_to_remove = []
        for meteor in self.meteors:
            meteor['y'] += meteor['speed']
            meteor['item'].setPos(meteor['x'], meteor['y'])
            
            # 检查是否超出屏幕
            if meteor['y'] > self.height():
                meteors_to_remove.append(meteor)
            
            # 检查是否与火箭碰撞
            elif rocket_rect and meteor['item'].sceneBoundingRect().intersects(rocket_rect):
                meteors_to_remove.append(meteor)
                
                # 扣分
                self.parent_widget.add_score(-20 * self.difficulty)
        
        # 删除超出屏幕或碰撞的陨石
        for meteor in meteors_to_remove:
            self.scene.removeItem(meteor['item'])
            self.meteors.remove(meteor)
            
    def resizeEvent(self, event):
        """处理窗口大小变化"""
        super().resizeEvent(event)
        
        # 更新场景尺寸
        if hasattr(self, 'scene'):
            self.scene.setSceneRect(0, 0, self.width(), self.height())
            
            # 更新火箭位置
            if self.rocket:
                rocket_y = self.height() * self.rocket_position
                self.rocket.setPos((self.width() - 60) / 2, rocket_y)


class MagicForestGame(BaseGame):
    """魔法森林大冒险游戏"""
    
    def __init__(self, parent):
        """初始化游戏"""
        super().__init__(parent)
        
        # TODO: 实现魔法森林游戏
        self.setStyleSheet("background-color: #003300;")
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("魔法森林大冒险游戏 - 敬请期待！", alignment=Qt.AlignCenter))


class ColorPuzzleGame(BaseGame):
    """色彩拼图奇遇游戏"""
    
    def __init__(self, parent):
        """初始化游戏"""
        super().__init__(parent)
        
        # TODO: 实现色彩拼图游戏
        self.setStyleSheet("background-color: #330033;")
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("色彩拼图奇遇游戏 - 敬请期待！", alignment=Qt.AlignCenter)) 