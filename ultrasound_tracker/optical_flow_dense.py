import cv2
import numpy as np
from .preprocessing import preprocess


class DenseFlowTracker:
    """
    Optical flow dense par algorithme de Farneback.

    Calcule un vecteur de déplacement (u,v) pour chaque pixel.
    Utile pour visualiser la déformation globale du tissu
    et comme source alternative dans le filtre de Kalman.
    """

    def __init__(self,
                 pyr_scale: float = 0.5,
                 levels: int = 3,
                 winsize: int = 15,
                 iterations: int = 3,
                 poly_n: int = 5,
                 poly_sigma: float = 1.2):

        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0
        )
        self.prev_gray = None

    def initialize(self, frame: np.ndarray):
        self.prev_gray = preprocess(frame)

    def compute(self, frame: np.ndarray):
        """
        Calcule le champ de flow dense entre la frame précédente
        et la frame courante.

        Retourne
        --------
        flow : np.ndarray (H, W, 2) — flow[y,x] = (u, v) en pixels
        """
        if self.prev_gray is None:
            self.initialize(frame)
            return np.zeros((*frame.shape, 2))

        gray = preprocess(frame)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.params
        )
        self.prev_gray = gray.copy()
        return flow

    def mean_displacement(self, frame: np.ndarray):
        """
        Retourne le déplacement moyen (dx, dy) sur toute l'image.
        Format prêt pour intégration dans Kalman.
        """
        flow = self.compute(frame)
        return flow.mean(axis=(0, 1))   # [mean_u, mean_v]

    def visualize(self, flow: np.ndarray):
        """
        Convertit le champ de flow en image HSV colorisée.
        Teinte = direction, Saturation = magnitude.
        """
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2   # teinte
        hsv[..., 1] = 255                       # saturation max
        hsv[..., 2] = cv2.normalize(            # valeur = magnitude
            mag, None, 0, 255, cv2.NORM_MINMAX
        )
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
