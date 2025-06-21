import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QMutex, QWaitCondition, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap


class DetectionWorker(QObject):
    finished = pyqtSignal()
    image_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

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

    def run(self):
        self.status_changed.emit("就绪")

        while self.running:
            # Wait for new task
            with QMutexLocker(self.mutex):
                if not self.restart and not self.abort:
                    self.status_changed.emit("等待输入...")
                    self.condition.wait(self.mutex)

                if self.abort:
                    self.abort = False
                    self.restart = False
                    continue

                self.restart = False
                self.currently_processing = True

            # Process the current task
            try:
                cap = None
                self.status_changed.emit(f"开始检测: {self.source_type}")

                if self.source_type == "camera":
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        self.error_occurred.emit("无法打开摄像头")
                        continue

                    while self.currently_processing:
                        with QMutexLocker(self.mutex):
                            if self.abort:
                                self.status_changed.emit("摄像头检测已停止")
                                break

                        ret, frame = cap.read()
                        if not ret:
                            self.error_occurred.emit("摄像头读取失败")
                            break

                        # Process frame
                        self.process_frame(frame)

                elif self.source_type == "video":
                    if not self.file_path:
                        continue

                    cap = cv2.VideoCapture(self.file_path)
                    if not cap.isOpened():
                        self.error_occurred.emit("无法打开视频文件")
                        continue

                    while self.currently_processing:
                        with QMutexLocker(self.mutex):
                            if self.abort:
                                self.status_changed.emit("视频检测已停止")
                                break

                        ret, frame = cap.read()
                        if not ret:
                            self.status_changed.emit("视频播放完成")
                            break

                        # Process frame
                        self.process_frame(frame)

                elif self.source_type == "image":
                    if not self.file_path:
                        continue

                    frame = cv2.imread(self.file_path)
                    if frame is None:
                        self.error_occurred.emit("无法读取图片文件")
                    else:
                        # Process image
                        self.process_frame(frame)
                        self.status_changed.emit("图片检测完成")

            except Exception as e:
                self.error_occurred.emit(f"处理错误: {str(e)}")
            finally:
                self.currently_processing = False
                if cap is not None:
                    cap.release()
                    cap = None

        self.finished.emit()

    def process_frame(self, frame):
        # Use YOLO model for detection
        results = self.model.predict(frame)
        annotated_frame = results[0].plot()

        # Convert to RGB
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Emit the result
        self.image_ready.emit(rgb_image)


class YOLODetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11 对象检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 加载YOLO模型
        self.model = YOLO(
            r"D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11FRFN\train2\weights\best.pt")

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

        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(300)
        self.status_label.setStyleSheet("font-weight: bold;")

        control_layout.addWidget(QLabel("输入源:"))
        control_layout.addWidget(self.source_combo)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        control_group.setLayout(control_layout)

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
        main_layout.addWidget(self.image_label)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

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
        self.update_status(f"错误: {message}")
        self.image_label.setText(message)
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