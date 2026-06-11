# ultrasound_tracker/kalman_fusion.py

import numpy as np
from filterpy.kalman import KalmanFilter


class FascicleKalman:
    """
    Filtre de Kalman pour la fusion KLT + Hough.

    État   : [angle, vitesse_angle, longueur, vitesse_longueur]
    Mesure : [angle_hough, longueur_hough]

    KLT pilote la prédiction (lisse, dérive).
    Hough fournit la mesure corrective (drift-free, bruité).
    Résultat : lisse ET drift-free.
    """

    def __init__(self,
                 dt: float = 1/30,
                 process_noise: float = 0.1,
                 measurement_noise: float = 2.0):

        self.dt = dt
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Matrice de transition F (modèle vitesse constante)
        self.kf.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ], dtype=float)

        # Matrice d'observation H
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=float)

        # Bruit de mesure R (bruit du Hough — à calibrer sur phantom)
        self.kf.R = np.eye(2) * measurement_noise ** 2

        # Bruit de processus Q (incertitude du modèle physique)
        self.kf.Q = np.eye(4) * process_noise ** 2

        # Covariance initiale P (grande incertitude au départ)
        self.kf.P = np.eye(4) * 100.0

        self.initialized = False

    def initialize(self, angle: float, length: float):
        """
        Initialise l'état avec la première mesure Hough valide.
        """
        self.kf.x = np.array([[angle],
                               [0.],
                               [length],
                               [0.]], dtype=float)
        self.initialized = True

    def predict(self, klt_displacement: np.ndarray = None):
        """
        Étape de prédiction.
        klt_displacement : [dx, dy] du KLT pour cette frame.
        KLT informe la vitesse estimée dans l'état prédit.
        """
        if not self.initialized:
            return

        # KLT informe la vitesse (dy → variation d'angle estimée)
        if klt_displacement is not None:
            dy = float(klt_displacement[1])
            self.kf.x[1, 0] = dy / self.dt   # vitesse angulaire

        self.kf.predict()

    def update(self, angle: float, length: float):
        """
        Étape de mise à jour avec la mesure Hough.
        """
        if not self.initialized:
            self.initialize(angle, length)
            return

        z = np.array([[angle], [length]], dtype=float)
        self.kf.update(z)

    def get_state(self):
        """
        Retourne l'état actuel fusionné.

        Retourne
        --------
        angle   : float — angle de pennation fusionné (°)
        length  : float — longueur de fascicule fusionnée (px)
        """
        return float(self.kf.x[0, 0]), float(self.kf.x[2, 0])

    def get_uncertainty(self):
        """
        Retourne l'incertitude courante (diagonale de P).
        Utile pour visualiser la confiance du filtre.
        """
        return float(self.kf.P[0, 0]), float(self.kf.P[2, 2])
