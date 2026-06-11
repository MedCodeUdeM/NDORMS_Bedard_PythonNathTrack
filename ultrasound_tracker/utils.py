import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def plot_results(results: dict, fps: float = 30.0,
                 title: str = "Tracking Results"):
    """
    Visualise les estimations Hough et Kalman sur le même graphe.
    """
    frames = np.array(results['frame'])
    time = frames / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Angle de pennation
    ax1.plot(time, results['hough_angle'],
             'r.', alpha=0.3, label='Hough (drift-free, bruité)')
    ax1.plot(time, results['kalman_angle'],
             'g-', linewidth=2, label='Kalman (fusionné)')
    ax1.set_ylabel('Angle de pennation (°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Longueur de fascicule
    ax2.plot(time, results['hough_length'],
             'r.', alpha=0.3, label='Hough')
    ax2.plot(time, results['kalman_length'],
             'g-', linewidth=2, label='Kalman')
    ax2.set_ylabel('Longueur de fascicule (px)')
    ax2.set_xlabel('Temps (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def save_results(results: dict, path: str):
    """
    Sauvegarde les résultats dans un fichier CSV.
    """
    if not PANDAS_AVAILABLE:
        print("pandas non installé — sauvegarde ignorée")
        return
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Résultats sauvegardés : {path}")


def draw_fascicles(frame: np.ndarray,
                   lines: np.ndarray,
                   color: tuple = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """
    Dessine les fascicules détectés sur une frame.
    """
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def draw_klt_points(frame: np.ndarray,
                    good_new: np.ndarray,
                    good_old: np.ndarray) -> np.ndarray:
    """
    Dessine les points KLT et leurs trajectoires.
    """
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if good_new is None:
        return vis
    for new, old in zip(good_new, good_old):
        x_new, y_new = new.ravel().astype(int)
        x_old, y_old = old.ravel().astype(int)
        cv2.arrowedLine(vis, (x_old, y_old),
                        (x_new, y_new), (0, 255, 0), 2)
        cv2.circle(vis, (x_new, y_new), 4, (0, 0, 255), -1)
    return vis