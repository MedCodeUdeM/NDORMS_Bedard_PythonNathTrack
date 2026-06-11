
import numpy as np
from scipy.signal import correlate2d


class SpeckleTracker:
    """
    Speckle tracking par corrélation croisée normalisée (NCC).

    Drift-free et peu bruité — exploite le pattern de speckle
    propre à chaque région tissulaire.
    Particulièrement adapté pour la fascia et les tissus mous
    sans structure linéaire claire (où Hough échoue).
    """

    def __init__(self,
                 kernel_size: int = 21,
                 search_radius: int = 15,
                 step: int = 10,
                 min_ncc: float = 0.5):

        self.kernel_size = kernel_size
        self.search_radius = search_radius
        self.step = step
        self.min_ncc = min_ncc   # seuil de confiance NCC

    def compute_displacement(self,
                             frame1: np.ndarray,
                             frame2: np.ndarray):
        """
        Calcule le champ de déplacement entre frame1 et frame2
        par NCC sur une grille de régions.

        Retourne
        --------
        disp_x     : champ de déplacement horizontal (pixels)
        disp_y     : champ de déplacement vertical (pixels)
        confidence : valeur NCC au pic (0→1) pour chaque région
        """
        h, w = frame1.shape
        hk = self.kernel_size // 2
        sr = self.search_radius

        disp_x = np.full((h, w), np.nan)
        disp_y = np.full((h, w), np.nan)
        confidence = np.zeros((h, w))

        for y in range(hk + sr, h - hk - sr, self.step):
            for x in range(hk + sr, w - hk - sr, self.step):

                # Pattern de référence dans frame1
                kernel = frame1[y-hk:y+hk+1,
                                x-hk:x+hk+1].astype(float)

                # Région de recherche dans frame2
                search = frame2[y-hk-sr:y+hk+sr+1,
                                x-hk-sr:x+hk+sr+1].astype(float)

                # Normalisation
                k_norm = kernel - kernel.mean()
                s_norm = search - search.mean()
                k_std = k_norm.std()
                s_std = s_norm.std()

                if k_std < 1e-6 or s_std < 1e-6:
                    continue   # région homogène, pas de texture

                # Corrélation croisée normalisée
                corr = correlate2d(s_norm, k_norm, mode='valid')
                corr /= (k_std * s_std * self.kernel_size ** 2)

                peak = np.unravel_index(np.argmax(corr), corr.shape)
                ncc_val = corr[peak]

                if ncc_val < self.min_ncc:
                    continue   # match pas assez confiant

                disp_y[y, x] = peak[0] - sr
                disp_x[y, x] = peak[1] - sr
                confidence[y, x] = ncc_val

        return disp_x, disp_y, confidence

    def mean_displacement(self,
                          frame1: np.ndarray,
                          frame2: np.ndarray):
        """
        Retourne le déplacement moyen (dx, dy) pondéré par la
        confiance NCC — format prêt pour intégration dans Kalman.
        """
        dx, dy, conf = self.compute_displacement(frame1, frame2)

        mask = ~np.isnan(dx) & (conf > self.min_ncc)
        if mask.sum() == 0:
            return np.array([0., 0.])

        weights = conf[mask]
        mean_dx = np.average(dx[mask], weights=weights)
        mean_dy = np.average(dy[mask], weights=weights)
        return np.array([mean_dx, mean_dy])