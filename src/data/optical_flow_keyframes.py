import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN

from src.utils.io import ensure_dir


class OpticalFlowKeyframeExtractor:
    def __init__(self, lambda_weight: float = 1.0, threshold_ratio: float = 0.3) -> None:
        self.lambda_weight = lambda_weight
        self.threshold_ratio = threshold_ratio
        self.face_detector = MTCNN()

    def _compute_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(rgb_frame)
        if not faces:
            return None
        main_face = max(faces, key=lambda x: x["confidence"])
        x, y, w, h = main_face["box"]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        x = max(0, x - margin_w)
        y = max(0, y - margin_h)
        w = min(frame.shape[1] - x, w + 2 * margin_w)
        h = min(frame.shape[0] - y, h + 2 * margin_h)
        return x, y, w, h

    def _calculate_weighted_energy(self, flow: np.ndarray, face_coords: Tuple[int, int, int, int]) -> float:
        x, y, w, h = face_coords
        flow_face = flow[y : y + h, x : x + w]
        u_x = flow_face[..., 0]
        u_y = flow_face[..., 1]
        velocity = np.sqrt(u_x**2 + u_y**2)
        angles = np.arctan2(u_y, u_x)
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)

        angle_avg = np.mean(angles)
        angle_max = angles[np.unravel_index(np.argmax(velocity), velocity.shape)]

        weight_term1 = (np.abs(angles - angle_avg) / np.pi) * self.lambda_weight
        weight_term2 = (np.abs(angles - angle_max) / np.pi) * self.lambda_weight
        weights = weight_term1**2 + weight_term2**2
        return float(np.sum(weights * velocity**2))

    def _visualize_optical_flow(self, flow: np.ndarray) -> np.ndarray:
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _crop_and_resize(self, image: np.ndarray, coords: Tuple[int, int, int, int], size=(224, 224)) -> np.ndarray:
        x, y, w, h = coords
        cropped = image[y : y + h, x : x + w]
        return cv2.resize(cropped, size)

    def extract_keyframes(
        self,
        video_path: str,
        output_flow_dir: Optional[str] = None,
    ) -> Tuple[List[int], List[float]]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            return [], []

        energies = []
        prev_frame = frames[0]
        prev_face = self._detect_face(prev_frame)
        for i in range(1, len(frames)):
            curr_frame = frames[i]
            curr_face = self._detect_face(curr_frame)
            if prev_face is None or curr_face is None:
                energies.append(0.0)
                prev_frame = curr_frame
                prev_face = curr_face
                continue
            flow = self._compute_optical_flow(prev_frame, curr_frame)
            energy = self._calculate_weighted_energy(flow, curr_face)
            energies.append(energy)
            prev_frame = curr_frame
            prev_face = curr_face

        max_energy = max(energies) if energies else 1.0
        threshold = self.threshold_ratio * max_energy
        keyframes = [i for i, e in enumerate(energies) if e >= threshold]

        if output_flow_dir:
            ensure_dir(output_flow_dir)
            for i in keyframes:
                if i >= len(frames):
                    continue
                if i > 0:
                    flow = self._compute_optical_flow(frames[i - 1], frames[i])
                    flow_img = self._visualize_optical_flow(flow)
                    face = self._detect_face(frames[i])
                    if face:
                        flow_face = self._crop_and_resize(flow_img, face)
                        cv2.imwrite(os.path.join(output_flow_dir, f"flow_face_{i}.png"), flow_face)

        return keyframes, energies
