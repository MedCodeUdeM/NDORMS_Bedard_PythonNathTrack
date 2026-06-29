import cv2
import numpy as np
from ultrasound_tracker.legacy.preprocessing import preprocess


class KLTTracker:
    """
    Tracker KLT (Kanade-Lucas-Tomasi) pour fascicules musculaires.

    Méthode séquentielle : lisse, peu bruité, mais dérive dans le temps.
    Utilisé comme source de prédiction dans le filtre de Kalman.
    """

    def __init__(self,
                 max_corners: int = 50,
                 quality_level: float = 0.3,
                 min_distance: int = 7,
                 win_size: tuple = (15, 15),
                 max_level: int = 3):

        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30, 0.01
            )
        )
        self.prev_gray = None
        self.prev_points = None
        self.initialized = False

    def initialize(self, frame: np.ndarray):
        """
        Initialise le tracker sur la première frame.
        Détecte les points d'intérêt (Shi-Tomasi).
        """
        gray = preprocess(frame)
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, mask=None, **self.feature_params
        )
        self.prev_gray = gray.copy()
        self.initialized = True
        n = len(self.prev_points) if self.prev_points is not None else 0
        print(f"KLT initialisé : {n} points détectés")

    def update(self, frame: np.ndarray):
        """
        Calcule le déplacement des points entre la frame précédente
        et la frame courante.

        Retourne
        --------
        mean_displacement : np.ndarray [dx, dy] en pixels
        good_new          : positions actuelles des points trackés
        good_old          : positions précédentes des points trackés
        """
        if not self.initialized:
            raise RuntimeError("Appelle initialize() d'abord.")

        gray = preprocess(frame)

        if self.prev_points is None or len(self.prev_points) == 0:
            # Plus de points — on réinitialise
            self.initialize(frame)
            return np.array([0., 0.]), None, None

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            **self.lk_params
        )

        good_new = new_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) == 0:
            self.initialize(frame)
            return np.array([0., 0.]), None, None

        # Déplacement moyen de tous les points trackés
        displacement = good_new - good_old
        mean_displacement = displacement.mean(axis=0)

        # Rafraîchir les points perdus si trop peu restent
        if len(good_new) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()
        return mean_displacement, good_new, good_old
