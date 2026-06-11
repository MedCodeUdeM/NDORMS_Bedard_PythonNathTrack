import cv2
import numpy as np
from .preprocessing import preprocess


class HoughDetector:
    """
    Détecteur de fascicules par transformée de Hough probabiliste.

    Méthode non-séquentielle : drift-free, mais jitter entre frames.
    Utilisé comme mesure dans le filtre de Kalman.
    """

    def __init__(self,
                 angle_min: float = 10.0,
                 angle_max: float = 40.0,
                 canny_low: int = 30,
                 canny_high: int = 90,
                 hough_threshold: int = 40,
                 min_line_length: int = 30,
                 max_line_gap: int = 10):

        self.angle_min = angle_min
        self.angle_max = angle_max
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect(self, frame: np.ndarray):
        """
        Détecte les fascicules dans une frame.

        Retourne
        --------
        lines   : np.ndarray (N, 4) — segments (x1,y1,x2,y2)
        angles  : np.ndarray (N,)   — angle de chaque segment (°)
        lengths : np.ndarray (N,)   — longueur de chaque segment (px)
        None, None, None si aucun fascicule détecté
        """
        enhanced = preprocess(frame, contrast=True, blur=True)

        # Détection de contours
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high,
                          apertureSize=3)

        # Hough transform probabiliste
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return None, None, None

        lines = lines[:, 0, :]   # shape (N, 4)

        # Filtrage angulaire
        filtered, angles, lengths = [], [], []

        for x1, y1, x2, y2 in lines:
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
            length = np.hypot(dx, dy)

            if self.angle_min <= angle <= self.angle_max:
                filtered.append([x1, y1, x2, y2])
                angles.append(angle)
                lengths.append(length)

        if not filtered:
            return None, None, None

        return (np.array(filtered),
                np.array(angles),
                np.array(lengths))

    def estimate(self, frame: np.ndarray):
        """
        Retourne l'estimation scalaire (médiane) de l'angle
        et de la longueur de fascicule — format prêt pour Kalman.

        Retourne
        --------
        (angle_median, length_median) ou (None, None)
        """
        lines, angles, lengths = self.detect(frame)
        if angles is None or len(angles) == 0:
            return None, None
        return float(np.median(angles)), float(np.median(lengths))
