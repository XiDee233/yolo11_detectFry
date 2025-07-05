import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import deque

def compute_iou(a, b):
    # a, b: [x, y, w, h]
    rect_a = [a[0], a[1], a[0]+a[2], a[1]+a[3]]
    rect_b = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    xx1 = max(rect_a[0], rect_b[0])
    yy1 = max(rect_a[1], rect_b[1])
    xx2 = min(rect_a[2], rect_b[2])
    yy2 = min(rect_a[3], rect_b[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])
    area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

class KalmanTracker:
    count = 0
    def __init__(self, bbox, id_=None, max_history=30, frame_w=480, frame_h=640):
        # bbox: [x1, y1, x2, y2]
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.id = KalmanTracker.count if id_ is None else id_
        KalmanTracker.count += 1
        self.age = 0
        self.lost = 0
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        # --------- 初始速度设置（仿C++） ---------
        relative_x = x / frame_w
        relative_y = y / frame_h
        vx, vy = 0.0, 0.0
        VERTICAL_SPEED = 50.0
        HORIZONTAL_SPEED = 50.0
        if relative_x < 0.3:
            vx = HORIZONTAL_SPEED
        elif relative_x > 0.7:
            vx = -HORIZONTAL_SPEED
        elif relative_y > 0.7:
            vy = -VERTICAL_SPEED
        elif relative_y < 0.3:
            vy = VERTICAL_SPEED
        # ----------------------------------------
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0.5,0],
            [0,1,0,0,0,1,0,0.5],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,1,0],
            [0,0,0,0,0,1,0,1],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((4,8), np.float32)
        self.kf.measurementMatrix[:4,:4] = np.eye(4)
        # --- processNoiseCov ---
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov[0,0] = 0.01
        self.kf.processNoiseCov[1,1] = 0.01
        self.kf.processNoiseCov[2,2] = 0.01
        self.kf.processNoiseCov[3,3] = 0.01
        self.kf.processNoiseCov[4,4] = 0.01
        self.kf.processNoiseCov[5,5] = 0.01
        self.kf.processNoiseCov[6,6] = 0.1
        self.kf.processNoiseCov[7,7] = 0.1
        # --- measurementNoiseCov ---
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.2
        # --- errorCovPost ---
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        # --- 状态初始化 ---
        self.kf.statePre = np.array([[x],[y],[w],[h],[vx],[vy],[0],[0]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        # 立即记录首帧轨迹点
        self.history.append((x, y))

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.lost += 1
        # 不在predict时记录轨迹

    def update(self, bbox):
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        measurement = np.array([[x],[y],[w],[h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.lost = 0
        self.update_history()  # 只在update时记录轨迹

    def update_history(self):
        x = float(self.kf.statePost[0])
        y = float(self.kf.statePost[1])
        self.history.append((x, y))

    def get_state(self):
        x = self.kf.statePost[0,0]
        y = self.kf.statePost[1,0]
        w = self.kf.statePost[2,0]
        h = self.kf.statePost[3,0]
        return [x-w/2, y-h/2, x+w/2, y+h/2]

    def get_rect(self):
        x = self.kf.statePost[0,0]
        y = self.kf.statePost[1,0]
        w = self.kf.statePost[2,0]
        h = self.kf.statePost[3,0]
        return [x-w/2, y-h/2, w, h]

class ObjectTracker:
    def __init__(self, max_lost=30):
        self.max_lost = max_lost
        self.trackers = []

    def update(self, dets, frame_shape=None):
        dets = [d[:4] for d in dets]
        N = len(self.trackers)
        M = len(dets)
        frame_h, frame_w = frame_shape[:2] if frame_shape is not None else (640, 480)
        max_distance = np.sqrt(frame_w**2 + frame_h**2)
        for trk in self.trackers:
            trk.predict()
        cost_matrix = np.ones((N, M), dtype=np.float32)
        if N > 0 and M > 0:
            for i, trk in enumerate(self.trackers):
                pred = trk.get_rect()
                pred_cx = pred[0] + pred[2]/2
                pred_cy = pred[1] + pred[3]/2
                for j, det in enumerate(dets):
                    iou = compute_iou(pred, [det[0], det[1], det[2]-det[0], det[3]-det[1]])
                    det_cx = (det[0] + det[2]) / 2
                    det_cy = (det[1] + det[3]) / 2
                    dx = det_cx - pred_cx
                    dy = det_cy - pred_cy
                    distance_cost = np.sqrt(dx*dx + dy*dy) / (max_distance * 0.2)
                    cost_matrix[i, j] = (1 - iou) * 0.3 + distance_cost * 0.7
        assignment = [-1]*N
        if N > 0 and M > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.8:
                    assignment[r] = c
        assigned_det = [False]*M
        for i, trk in enumerate(self.trackers):
            if assignment[i] != -1:
                trk.update(dets[assignment[i]])
                assigned_det[assignment[i]] = True
        for j, flag in enumerate(assigned_det):
            if not flag:
                self.trackers.append(KalmanTracker(dets[j], frame_w=frame_w, frame_h=frame_h))
        remain = []
        results = []
        for trk in self.trackers:
            if trk.lost < self.max_lost:
                remain.append(trk)
                x1, y1, x2, y2 = trk.get_state()
                results.append({
                    'id': trk.id,
                    'bbox': (x1, y1, x2, y2),
                    'trace': list(trk.history)
                })
        self.trackers = remain
        return results

    def draw_tracks(self, frame):
        for trk in self.trackers:
            if len(trk.history) > 1:
                pts = trk.history
                for i in range(1, len(pts)):
                    cv2.line(frame, (int(pts[i-1][0]), int(pts[i-1][1])), (int(pts[i][0]), int(pts[i][1])), (0,255,0), 2)
        return frame 