import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import deque


def compute_iou(a, b):
    """计算检测框与跟踪框的交并比"""
    # a, b: [x1, y1, x2, y2]
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    a_area = (xa2 - xa1) * (ya2 - ya1)
    b_area = (xb2 - xb1) * (yb2 - yb1)
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def compute_euclidean_distance(a, b):
    """计算两检测框中心点的欧氏距离"""
    # a, b: [x1, y1, x2, y2]
    center_a = [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]
    center_b = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    return np.sqrt((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2)


def compute_diou(a, b, frame_w, frame_h):
    """计算Distance-IoU（DIoU）"""
    iou = compute_iou(a, b)
    if iou <= 0:
        return 0

    # 中心点距离
    center_a = [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]
    center_b = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    dist_sq = dx * dx + dy * dy

    # 最小包围矩形对角线距离
    c_w = max(a[2], b[2]) - min(a[0], b[0])
    c_h = max(a[3], b[3]) - min(a[1], b[1])
    c_dist_sq = c_w * c_w + c_h * c_h

    return iou - (dist_sq / c_dist_sq)


class KalmanTracker:
    """基于卡尔曼滤波的目标跟踪器，支持轨迹恢复"""
    count = 0

    def __init__(self, bbox, id_=None, max_history=30, frame_w=640, frame_h=480):
        # bbox: [x1, y1, x2, y2]
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.id = KalmanTracker.count if id_ is None else id_
        KalmanTracker.count += 1
        self.age = 0
        self.lost = 0
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.last_detection = bbox  # 保存最后一次成功匹配的检测框
        self.frame_w, self.frame_h = frame_w, frame_h
        self.predict_state = None  # 添加：存储预测状态但不更新

        # 卡尔曼滤波初始化（8维状态：x,y,w,h,vx,vy,ax,ay）
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 0, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        # 过程噪声与测量噪声设置
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:6, 4:6] = np.eye(2, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.2
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.9

        # 状态初始化（包含初始速度估计）
        self.kf.statePre = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        self.update_history()  # 记录首帧轨迹

    def predict(self):
        """预测下一帧状态，但不更新内部状态"""
        # 修改：返回预测状态但不更新内部状态
        state = self.kf.predict().copy()
        x, y = state[0, 0], state[1, 0]
        w, h = state[2, 0], state[3, 0]
        self.predict_state = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        self.age += 1
        return self.predict_state

    def update(self, bbox):
        """用检测框更新跟踪状态"""
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.last_detection = bbox  # 更新最后检测
        self.lost = 0
        self.update_history()  # 记录轨迹

    def update_history(self):
        """更新轨迹历史"""
        x = float(self.kf.statePost[0])
        y = float(self.kf.statePost[1])
        self.history.append((x, y))

    def get_state(self):
        """获取当前跟踪框"""
        x, y = self.kf.statePost[0, 0], self.kf.statePost[1, 0]
        w, h = self.kf.statePost[2, 0], self.kf.statePost[3, 0]
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    def get_last_detection(self):
        """获取最后一次检测框（用于轨迹恢复）"""
        return self.last_detection


class ObjectTracker:
    """多目标跟踪器，实现论文中的改进SORT算法"""

    def __init__(self, max_lost=30, score_thresh=0.7):
        self.max_lost = max_lost
        self.score_thresh = score_thresh
        self.trackers = []
        self.next_id = 1  # 简化ID管理，只使用递增ID
        self.id_mapping = {}  # 添加ID映射字典初始化
        self.match_threshold = 0.3  # 匹配阈值，论文中使用0.3
        # 添加最小匹配距离阈值，防止远距离错误匹配
        self.max_distance_factor = 0.1  # 最大距离因子，相对于图像对角线
        self.decay_factor = 0.9  # 新增：时间衰减因子

    def update(self, dets, frame_shape=None):
        """
        输入:
            dets: 检测框列表，格式为[[x1,y1,x2,y2,score], ...]
            frame_shape: 帧尺寸(h, w)或(h, w, c)，用于计算DIoU
        输出:
            跟踪结果列表，每个元素包含id、bbox和轨迹
        """
        if len(dets) == 0:
            # 无检测时仅预测轨迹
            for trk in self.trackers:
                trk.predict()
                trk.lost += 1  # 修改：在无检测时增加丢失计数
            return self._get_tracking_results()

        # 分离检测框和分数
        det_boxes = [d[:4] for d in dets]
        det_scores = [d[4] for d in dets]
        # 修改这一行，只取前两个值
        if frame_shape is not None:
            frame_h, frame_w = frame_shape[:2]  # 只取高度和宽度，忽略通道数
        else:
            frame_h, frame_w = 480, 640
        max_distance = np.sqrt(frame_w ** 2 + frame_h ** 2) * self.max_distance_factor

        # === 修正1：严格遵循预测->匹配->更新的时序 ===
        # 步骤1：预测所有轨迹状态（但不更新历史）
        pred_states = [trk.predict() for trk in self.trackers]  # 返回预测状态不写入

        # 第二步：将检测分为高分(>=thresh)和低分(<thresh)
        high_score_idx = [i for i, s in enumerate(det_scores) if s >= self.score_thresh]
        low_score_idx = [i for i, s in enumerate(det_scores) if s < self.score_thresh]
        high_dets = [det_boxes[i] for i in high_score_idx]
        low_dets = [det_boxes[i] for i in low_score_idx]

        # 第三步：高分检测与轨迹的第一阶段匹配（基于欧氏距离变异体D_ij）
        high_match_results = self._first_matching(self.trackers, high_dets, frame_w, frame_h, max_distance)
        high_tracked_indices, high_matched_dets = high_match_results
        
        # 第四步：低分检测与未匹配轨迹的第二阶段匹配（基于DIoU_2）
        unmatched_trackers = [i for i in range(len(self.trackers)) if i not in high_tracked_indices]
        low_match_results = self._second_matching(
            [self.trackers[i] for i in unmatched_trackers],
            low_dets, frame_w, frame_h, max_distance
        )
        
        # 将未匹配轨迹的索引映射回原始轨迹索引
        low_tracked_indices = [unmatched_trackers[i] for i in low_match_results[0]]
        low_matched_dets = low_match_results[1]
        
        # 合并两阶段匹配结果
        tracked_indices = high_tracked_indices + low_tracked_indices
        
        # 第五步：轨迹恢复（严重遮挡时用最后检测匹配）
        unmatched_tracks = [i for i in range(len(self.trackers)) if i not in tracked_indices]
        unmatched_dets = [i for i in range(len(det_boxes)) if i not in high_matched_dets + low_matched_dets]
        
        # 创建已分配检测的标记数组
        assigned_det = [False] * len(det_boxes)
        # 标记已经在第一阶段和第二阶段匹配的检测
        for i in high_matched_dets:
            assigned_det[high_score_idx[i]] = True
        for i in low_matched_dets:
            assigned_det[low_score_idx[i]] = True
        
        # 轨迹恢复匹配
        recovered_track_indices = []
        if len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            recovered_tracks = self._trajectory_recovery(
                [self.trackers[i] for i in unmatched_tracks],
                [det_boxes[i] for i in unmatched_dets],
                frame_w, frame_h, max_distance
            )
            
            # 直接更新恢复的轨迹，并标记检测已分配
            for trk, det_idx in recovered_tracks:
                trk_idx = self.trackers.index(trk)
                actual_det_idx = unmatched_dets[det_idx]
                
                # 只有当检测未被分配时才更新轨迹
                if not assigned_det[actual_det_idx]:
                    self.trackers[trk_idx].update(det_boxes[actual_det_idx])
                    assigned_det[actual_det_idx] = True
                    recovered_track_indices.append(trk_idx)
        
        # 第六步：更新匹配的轨迹，添加新轨迹
        for i, trk in enumerate(self.trackers):
            if i in high_tracked_indices:
                # 高分检测匹配
                high_idx = high_tracked_indices.index(i)
                det_idx = high_score_idx[high_matched_dets[high_idx]]
                trk.update(det_boxes[det_idx])
            elif i in low_tracked_indices:
                # 低分检测匹配
                low_idx = low_tracked_indices.index(i)
                det_idx = low_score_idx[low_matched_dets[low_idx]]
                trk.update(det_boxes[det_idx])
            elif i in recovered_track_indices:
                # 轨迹已在恢复阶段更新，无需再次更新
                pass
            else:
                # 未匹配则增加丢失计数
                trk.lost += 1
        
        # 添加新检测为新轨迹
        for i, flag in enumerate(assigned_det):
            if not flag:
                new_trk = KalmanTracker(det_boxes[i], id_=self.next_id, frame_w=frame_w, frame_h=frame_h)
                self.next_id += 1
                self.trackers.append(new_trk)
        
        # 更新ID映射
        for id in list(self.id_mapping.keys()):
            if any(trk.id == id for trk in self.trackers):
                self.id_mapping[id] = 0  # 活跃
            else:
                self.id_mapping[id] = self.id_mapping.get(id, 0) - 1  # 递减计数

        # 过滤长时间丢失的轨迹
        self.trackers = [trk for trk in self.trackers if trk.lost < self.max_lost]
        return self._get_tracking_results()

    def _first_matching(self, trackers, detections, frame_w, frame_h, max_distance):
        """第一阶段匹配：高分检测与轨迹基于欧氏距离变异体D_ij匹配"""
        N, M = len(trackers), len(detections)
        if N == 0 or M == 0:
            return [], []

        cost_matrix = np.ones((N, M), dtype=np.float32) * 2.0
        for i, trk in enumerate(trackers):
            # 修改：使用预测状态而非当前状态
            pred_box = trk.predict_state if trk.predict_state is not None else trk.get_state()
            pred_center = [(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2]
            for j, det in enumerate(detections):
                det_center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
                dx = det_center[0] - pred_center[0]
                dy = det_center[1] - pred_center[1]
                # 计算绝对距离，用于过滤远距离匹配
                abs_dist = np.sqrt(dx * dx + dy * dy)
                if abs_dist > max_distance:
                    continue  # 跳过远距离匹配
                
                # 计算D_ij = 1 - (欧氏距离^2 / 对角线距离^2)
                dist_sq = dx * dx + dy * dy
                c_sq = frame_w * frame_w + frame_h * frame_h
                d_ij = 1.0 - (dist_sq / c_sq)
                cost_matrix[i, j] = 1.0 - d_ij  # 代价越小越匹配

        # 匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.match_threshold:  # 使用论文中的阈值0.3
                valid_matches.append((r, c))

        tracked_indices = [r for r, c in valid_matches]
        matched_dets = [c for r, c in valid_matches]
        return tracked_indices, matched_dets

    def _second_matching(self, trackers, detections, frame_w, frame_h, max_distance):
        """第二阶段匹配：低分检测与未匹配轨迹基于DIoU_2匹配"""
        N, M = len(trackers), len(detections)
        if N == 0 or M == 0:
            return [], []

        cost_matrix = np.ones((N, M), dtype=np.float32) * 2.0
        for i, trk in enumerate(trackers):
            # 修改：使用预测状态而非当前状态
            pred_box = trk.predict_state if trk.predict_state is not None else trk.get_state()
            pred_center = [(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2]
            for j, det in enumerate(detections):
                # 计算绝对距离，用于过滤远距离匹配
                det_center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
                dx = det_center[0] - pred_center[0]
                dy = det_center[1] - pred_center[1]
                abs_dist = np.sqrt(dx * dx + dy * dy)
                if abs_dist > max_distance:
                    continue  # 跳过远距离匹配
                
                # 计算DIoU_2 = (IoU + D_ij) / 2
                iou = compute_iou(pred_box, det)
                
                # 计算D_ij
                dist_sq = dx * dx + dy * dy
                c_sq = frame_w * frame_w + frame_h * frame_h
                d_ij = 1.0 - (dist_sq / c_sq)
                
                # DIoU_2
                diou_2 = (iou + d_ij) / 2.0
                cost_matrix[i, j] = 1.0 - diou_2  # 代价越小越匹配

        # 匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.match_threshold:  # 使用论文中的阈值0.3
                valid_matches.append((r, c))

        tracked_indices = [r for r, c in valid_matches]
        matched_dets = [c for r, c in valid_matches]
        return tracked_indices, matched_dets

    def _trajectory_recovery(self, trackers, detections, frame_w, frame_h, max_distance):
        """轨迹恢复：用最后检测框匹配严重遮挡的轨迹"""
        N, M = len(trackers), len(detections)
        if N == 0 or M == 0:
            return []

        cost_matrix = np.ones((N, M), dtype=np.float32) * 2.0
        
        for i, trk in enumerate(trackers):
            last_det = trk.get_last_detection()
            
            for j, det in enumerate(detections):
                # 计算绝对距离，用于过滤远距离匹配
                last_center = [(last_det[0] + last_det[2]) / 2, (last_det[1] + last_det[3]) / 2]
                det_center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
                dx = det_center[0] - last_center[0]
                dy = det_center[1] - last_center[1]
                abs_dist = np.sqrt(dx * dx + dy * dy)
                if abs_dist > max_distance:
                    continue  # 跳过远距离匹配
                
                # 计算最后检测与当前检测的DIoU_2
                iou = compute_iou(last_det, det)
                
                # 计算D_ij
                dist_sq = dx * dx + dy * dy
                c_sq = frame_w * frame_w + frame_h * frame_h
                d_ij = 1.0 - (dist_sq / c_sq)
                
                # 添加时间衰减因子 (1 - decay_factor^lost)
                time_decay = 1.0 - (self.decay_factor ** trk.lost)
                
                # DIoU_2 (添加时间衰减因子)
                diou_2 = (iou + d_ij) / 2.0 * time_decay
                cost_matrix[i, j] = 1.0 - diou_2  # 代价越小越匹配

        # 匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            # 使用与其他匹配相同的阈值，论文中未提及使用更严格的阈值
            if cost_matrix[r, c] < self.match_threshold:
                valid_matches.append((trackers[r], c))
        return valid_matches

    def _get_tracking_results(self):
        """获取跟踪结果列表"""
        results = []
        for trk in self.trackers:
            x1, y1, x2, y2 = trk.get_state()
            results.append({
                'id': trk.id,
                'bbox': (x1, y1, x2, y2),
                'trace': list(trk.history)
            })
        return results

    def draw_tracks(self, frame):
        """绘制轨迹"""
        for trk in self.trackers:
            if len(trk.history) > 1:
                for i in range(1, len(trk.history)):
                    pt1 = (int(trk.history[i - 1][0]), int(trk.history[i - 1][1]))
                    pt2 = (int(trk.history[i][0]), int(trk.history[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        return frame