# 在文件顶部导入日志模块
import sys
import cv2
import numpy as np
import logging  # 添加日志模块
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QMutex, QWaitCondition, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
from tracker import ObjectTracker
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detection_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YOLODetection")


class DetectionWorker(QObject):
    finished = pyqtSignal()
    image_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    # 在DetectionWorker类中添加一个新属性来跟踪当前检测到的ID总数

    # 在DetectionWorker类的__init__方法中添加新的计数器属性
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.source_type = None
        self.file_path = None
        self.running = True
        self.restart = False
        self.abort = False
        self.currently_processing = False
        self.tracker = ObjectTracker(max_lost=4)
        self.id_colors = {}
        self.show_inactive = True
        self.paused = False
        
        # 计数线相关属性
        self.counting_line = None  # 将在第一帧初始化
        self.line_position = 0.5  # 默认在画面中间
        self.line_direction = 'horizontal'  # 水平线
        self.crossed_ids = set()  # 已经穿过线的ID集合
        self.counter = 0  # 计数器
        self.total_ids = set()  # 跟踪所有出现过的ID
        
        # 添加双向计数属性
        self.left_to_right_counter = 0  # 从左到右计数
        self.right_to_left_counter = 0  # 从右到左计数
        self.top_to_bottom_counter = 0  # 从上到下计数
        self.bottom_to_top_counter = 0  # 从下到上计数
        self.crossed_direction = {}  # 记录每个ID穿过的方向

    def set_line_position(self, position):
        """设置计数线位置 (0-1 范围内的相对位置)"""
        with QMutexLocker(self.mutex):
            self.line_position = max(0.0, min(1.0, position))

    def set_line_direction(self, direction):
        """设置计数线方向 ('horizontal' 或 'vertical')"""
        with QMutexLocker(self.mutex):
            # 只有当方向确实改变时才重置计数线
            if self.line_direction != direction:
                self.line_direction = direction
                self.counting_line = None  # 重置计数线，使其在下一帧重新初始化

    # 修改reset_counter方法，重置所有计数器
    def reset_counter(self):
        """重置计数器"""
        with QMutexLocker(self.mutex):
            self.counter = 0
            self.left_to_right_counter = 0
            self.right_to_left_counter = 0
            self.top_to_bottom_counter = 0
            self.bottom_to_top_counter = 0
            self.crossed_ids = set()
            self.crossed_direction = {}

    def set_source(self, source_type, file_path=None):
        with QMutexLocker(self.mutex):
            self.source_type = source_type
            self.file_path = file_path
            self.restart = True
            self.abort = False
            self.condition.wakeOne()
            self.status_changed.emit(f"准备检测: {source_type}")

    def stop(self):
        with QMutexLocker(self.mutex):
            self.abort = True
            self.condition.wakeOne()
            self.status_changed.emit("已停止")

    def pause(self):
        with QMutexLocker(self.mutex):
            self.paused = True
            self.status_changed.emit("已暂停")

    def resume(self):
        with QMutexLocker(self.mutex):
            self.paused = False
            self.condition.wakeOne()
            self.status_changed.emit("已恢复")

    def run(self):
        self.status_changed.emit("就绪")
        cap = None
        while self.running:
            with QMutexLocker(self.mutex):
                if self.paused:
                    self.status_changed.emit("已暂停，等待恢复...")
                    self.condition.wait(self.mutex)
                if not self.restart and not self.abort:
                    self.status_changed.emit("等待输入...")
                    self.condition.wait(self.mutex)
                if self.abort:
                    self.abort = False
                    self.restart = False
                    if cap is not None:
                        cap.release()
                        cap = None
                    continue
                if self.restart:
                    if cap is not None:
                        cap.release()
                        cap = None
                self.currently_processing = True
                self.restart = False

            try:
                self.status_changed.emit(f"开始检测: {self.source_type}")
                if self.source_type == "camera":
                    if cap is None:
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            error_msg = "无法打开摄像头"
                            logger.error(error_msg)  # 记录到日志
                            self.error_occurred.emit(error_msg)
                            continue
                    while self.currently_processing and self.running:
                        with QMutexLocker(self.mutex):
                            if self.paused:
                                self.status_changed.emit("已暂停，等待恢复...")
                                self.condition.wait(self.mutex)
                            if self.abort or self.restart:
                                break
                        ret, frame = cap.read()
                        if not ret:
                            self.error_occurred.emit("摄像头读取失败")
                            break
                        self.process_frame(frame)
                elif self.source_type == "video":
                    if not self.file_path:
                        continue
                    if cap is None:
                        cap = cv2.VideoCapture(self.file_path)
                        if not cap.isOpened():
                            error_msg = "无法打开视频文件"
                            logger.error(error_msg)  # 记录到日志
                            self.error_occurred.emit(error_msg)
                            continue
                    while self.currently_processing and self.running:
                        with QMutexLocker(self.mutex):
                            if self.paused:
                                self.status_changed.emit("已暂停，等待恢复...")
                                self.condition.wait(self.mutex)
                            if self.abort or self.restart:
                                break
                        ret, frame = cap.read()
                        if not ret:
                            self.status_changed.emit("视频播放完成")
                            break
                        self.process_frame(frame)
                elif self.source_type == "image":
                    if not self.file_path:
                        continue
                    frame = cv2.imread(self.file_path)
                    if frame is None:
                        self.error_occurred.emit("无法读取图片文件")
                    else:
                        self.process_frame(frame)
                        self.status_changed.emit("图片检测完成")
            except Exception as e:
                error_msg = f"处理错误: {str(e)}"
                logger.exception(error_msg)  # 记录详细的异常信息到日志
                self.error_occurred.emit(error_msg)
            finally:
                self.currently_processing = False
        if cap is not None:
            cap.release()
            cap = None
        self.finished.emit()

    def get_color(self, track_id):
        if track_id not in self.id_colors:
            random.seed(track_id)
            self.id_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.id_colors[track_id]

    # 修改check_line_crossing方法，添加方向检测
    def check_line_crossing(self, track):
        """检查轨迹是否穿过计数线，并判断穿过方向"""
        # 如果计数线未初始化，直接返回False
        if self.counting_line is None:
            return False
            
        if track['id'] in self.crossed_ids:
            return False
            
        trace = track['trace']
        if len(trace) < 2:
            return False
            
        # 获取最近的两个点
        p1 = trace[-2]
        p2 = trace[-1]
        
        try:
            if self.line_direction == 'horizontal':
                # 水平线，检查是否从上到下或从下到上穿过
                line_y = self.counting_line[0][1]  # 线的y坐标
                if p1[1] < line_y and p2[1] >= line_y:
                    # 从上到下穿过
                    self.crossed_ids.add(track['id'])
                    self.crossed_direction[track['id']] = 'top_to_bottom'
                    self.top_to_bottom_counter += 1
                    return True
                elif p1[1] >= line_y and p2[1] < line_y:
                    # 从下到上穿过
                    self.crossed_ids.add(track['id'])
                    self.crossed_direction[track['id']] = 'bottom_to_top'
                    self.bottom_to_top_counter += 1
                    return True
            else:  # vertical
                # 垂直线，检查是否从左到右或从右到左穿过
                line_x = self.counting_line[0][0]  # 线的x坐标
                if p1[0] < line_x and p2[0] >= line_x:
                    # 从左到右穿过
                    self.crossed_ids.add(track['id'])
                    self.crossed_direction[track['id']] = 'left_to_right'
                    self.left_to_right_counter += 1
                    return True
                elif p1[0] >= line_x and p2[0] < line_x:
                    # 从右到左穿过
                    self.crossed_ids.add(track['id'])
                    self.crossed_direction[track['id']] = 'right_to_left'
                    self.right_to_left_counter += 1
                    return True
        except Exception as e:
            logger.error(f"计数线检测错误: {str(e)}")  # 记录到日志而不是显示在画面上
            # 不抛出异常，继续执行
                
        return False

    # 修改DetectionWorker类中的方法
    
    def process_frame(self, frame):
        if self.paused:
            return
            
        # 初始化计数线（如果尚未初始化）
        if self.counting_line is None:
            h, w = frame.shape[:2]
            if self.line_direction == 'horizontal':
                y = int(h * self.line_position)
                self.counting_line = [(0, y), (w, y)]
            else:  # vertical
                x = int(w * self.line_position)
                self.counting_line = [(x, 0), (x, h)]
                
        # Use YOLO model for detection
        results = self.model.predict(frame, conf=0.5, iou=0.7)
        boxes = results[0].boxes
        dets = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 0
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                dets.append([x1, y1, x2, y2, conf, cls])
        # 更新tracker
        tracks = self.tracker.update(dets, frame_shape=frame.shape)
        
        # 检查是否有物体穿过计数线
        for track in tracks:
            # 添加ID到总ID集合
            self.total_ids.add(track['id'])
            if self.check_line_crossing(track):
                self.counter += 1
                
        # 绘制检测框和轨迹（每个id唯一颜色）
        annotated_frame = frame.copy()
        for i, track in enumerate(tracks):
            # 判断活跃性
            trk_obj = self.tracker.trackers[i] if i < len(self.tracker.trackers) else None
            is_active = (trk_obj.lost == 0) if trk_obj else True
            if self.show_inactive or is_active:
                x1, y1, x2, y2 = map(int, track['bbox'])
                color = self.get_color(track['id'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID:{track['id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # 画轨迹（每个id唯一颜色）
        for i, track in enumerate(tracks):
            trk_obj = self.tracker.trackers[i] if i < len(self.tracker.trackers) else None
            is_active = (trk_obj.lost == 0) if trk_obj else True
            if self.show_inactive or is_active:
                color = self.get_color(track['id'])
                trace = track['trace']
                if len(trace) > 1:
                    for j in range(1, len(trace)):
                        pt1 = (int(trace[j-1][0]), int(trace[j-1][1]))
                        pt2 = (int(trace[j][0]), int(trace[j][1]))
                        cv2.line(annotated_frame, pt1, pt2, color, 2)
                        
        # 绘制计数线（确保计数线已初始化）
        if self.counting_line is not None:
            line_color = (0, 0, 255)  # 红色
            cv2.line(annotated_frame, self.counting_line[0], self.counting_line[1], line_color, 2)
        
        # 显示计数和总ID数
        count_text = f"Total Count: {self.counter}"
        cv2.putText(annotated_frame, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # 显示当前检测到的总ID数
        total_ids_text = f"Total IDs: {len(self.total_ids)}"
        cv2.putText(annotated_frame, total_ids_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 显示双向计数结果
        if self.line_direction == 'horizontal':
            # 显示从上到下和从下到上的计数
            top_to_bottom_text = f"Top→Bottom: {self.top_to_bottom_counter}"
            bottom_to_top_text = f"Bottom→Top: {self.bottom_to_top_counter}"
            cv2.putText(annotated_frame, top_to_bottom_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(annotated_frame, bottom_to_top_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        else:  # vertical
            # 显示从左到右和从右到左的计数
            left_to_right_text = f"Left→Right: {self.left_to_right_counter}"
            right_to_left_text = f"Right→Left: {self.right_to_left_counter}"
            cv2.putText(annotated_frame, left_to_right_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(annotated_frame, right_to_left_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        # 转为RGB
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        self.image_ready.emit(rgb_image)


class YOLODetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11 对象检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 加载YOLO模型
        self.model = YOLO(
            r"D:\pyProject\ultralytics_custom\runs\fish\detect\Archie\yolo11FRFN\train2\weights\best.pt")

        # 创建检测线程和工作器
        self.thread = QThread()
        self.worker = DetectionWorker(self.model)
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.image_ready.connect(self.update_image)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.status_changed.connect(self.update_status)

        # 启动线程
        self.thread.start()

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.is_paused = False

        self.init_ui()

    def init_ui(self):
        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头", "视频文件", "图片文件"])
        self.source_combo.currentIndexChanged.connect(self.reset_ui)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.start_detection)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_detection)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)

        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(300)
        self.status_label.setStyleSheet("font-weight: bold;")

        control_layout.addWidget(QLabel("输入源:"))
        control_layout.addWidget(self.source_combo)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        control_group.setLayout(control_layout)

        # 计数线控制面板
        line_control_group = QGroupBox("计数线控制")
        line_control_layout = QHBoxLayout()

        # 线位置滑块
        self.line_position_slider = QSlider(Qt.Horizontal)
        self.line_position_slider.setMinimum(0)
        self.line_position_slider.setMaximum(100)
        self.line_position_slider.setValue(50)  # 默认在中间
        self.line_position_slider.valueChanged.connect(self.update_line_position)

        # 线方向选择
        self.line_direction_combo = QComboBox()
        self.line_direction_combo.addItems(["水平线", "垂直线"])
        self.line_direction_combo.currentIndexChanged.connect(self.update_line_direction)

        # 重置计数按钮
        self.reset_counter_btn = QPushButton("重置计数")
        self.reset_counter_btn.clicked.connect(self.reset_counter)

        line_control_layout.addWidget(QLabel("线位置:"))
        line_control_layout.addWidget(self.line_position_slider)
        line_control_layout.addWidget(QLabel("线方向:"))
        line_control_layout.addWidget(self.line_direction_combo)
        line_control_layout.addWidget(self.reset_counter_btn)
        line_control_group.setLayout(line_control_layout)

        # 显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setText("等待输入...")
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #2C2C2C; 
                color: white; 
                font-size: 24px;
                border: 2px solid #444444;
                border-radius: 5px;
            }
        """)

        # 添加到主布局
        main_layout.addWidget(control_group)
        main_layout.addWidget(line_control_group)
        main_layout.addWidget(self.image_label)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def update_line_position(self):
        """更新计数线位置"""
        position = self.line_position_slider.value() / 100.0
        self.worker.set_line_position(position)

    def update_line_direction(self):
        """更新计数线方向"""
        direction = 'horizontal' if self.line_direction_combo.currentText() == "水平线" else 'vertical'
        self.worker.set_line_direction(direction)
        # 重置计数线，使其在下一帧重新初始化
        self.worker.counting_line = None

    def reset_counter(self):
        """重置计数器"""
        self.worker.reset_counter()

    def reset_ui(self):
        """当选择新的输入源时重置UI"""
        self.image_label.setText("等待输入...")
        self.image_label.setPixmap(QPixmap())
        self.update_status("就绪")

    def start_detection(self):
        source_type = self.source_combo.currentText()
        file_path = None

        if source_type == "视频文件":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
            if not file_path:
                return
            source_type = "video"

        elif source_type == "图片文件":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", "", "图片文件 (*.jpg *.png *.bmp *.jpeg)")
            if not file_path:
                return
            source_type = "image"

        else:  # 摄像头
            source_type = "camera"

        # 设置新的检测任务
        self.worker.set_source(source_type, file_path)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_detection(self):
        self.worker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(np.ndarray)
    def update_image(self, rgb_image):
        """更新显示图像"""
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # 创建QImage
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        if not q_img.isNull():
            pixmap = QPixmap.fromImage(q_img)
            # 自适应缩放保持宽高比
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("无法显示图像")

    @pyqtSlot(str)
    def show_error(self, message):
        """显示错误信息"""
        # 记录到日志
        logger.error(message)
        # 更新状态栏而不是在画面上显示
        self.update_status(f"错误: {message}")
        # 不再在图像标签上显示错误信息
        # self.image_label.setText(message)  # 注释掉这行
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(str)
    def update_status(self, message):
        """更新状态标签"""
        self.status_label.setText(message)

        # 添加状态颜色指示
        if "错误" in message:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        elif "检测" in message:
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")

    def toggle_pause(self):
        if not self.is_paused:
            self.worker.pause()
            self.pause_btn.setText("恢复")
            self.is_paused = True
        else:
            self.worker.resume()
            self.pause_btn.setText("暂停")
            self.is_paused = False

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        self.worker.running = False
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLODetectionApp()
    window.show()
    sys.exit(app.exec_())