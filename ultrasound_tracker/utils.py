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

# ============================================================================
# DRAWING ON EXISTING IMAGE
# ============================================================================

def draw_line_on_image(image: np.ndarray,
                       line,
                       color: tuple = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    Draw one line [x1, y1, x2, y2] on an existing BGR image.

    This differs from draw_fascicles(), which creates a new BGR image from
    a grayscale frame. Here, the input image is modified in-place and returned.
    """
    if line is None:
        return image

    x1, y1, x2, y2 = np.asarray(line).astype(int)

    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    return image


def draw_lines_on_image(image: np.ndarray,
                        lines,
                        color: tuple = (0, 255, 0),
                        thickness: int = 1) -> np.ndarray:
    """
    Draw multiple lines on an existing BGR image.

    Parameters
    ----------
    image : np.ndarray
        Existing BGR image.
    lines : np.ndarray shape (N, 4) or None
    color : tuple
        BGR color.
    thickness : int

    Returns
    -------
    image : np.ndarray
    """
    if lines is None:
        return image

    for line in lines:
        draw_line_on_image(
            image,
            line,
            color=color,
            thickness=thickness,
        )

    return image


def draw_points_on_image(image: np.ndarray,
                         points,
                         color: tuple = (0, 0, 255),
                         radius: int = 5) -> np.ndarray:
    """
    Draw points [[x, y], ...] on an existing BGR image.

    Parameters
    ----------
    image : np.ndarray
        Existing BGR image.
    points : np.ndarray shape (N, 2) or None
    color : tuple
        BGR color.
    radius : int

    Returns
    -------
    image : np.ndarray
    """
    if points is None:
        return image

    points = np.asarray(points, dtype=np.float32)

    for point in points:
        if np.all(np.isfinite(point)):
            x, y = point.astype(int)
            cv2.circle(
                image,
                (x, y),
                radius,
                color,
                -1,
            )

    return image


def put_text_lines_on_image(image: np.ndarray,
                            text_lines,
                            origin: tuple = (30, 40),
                            line_spacing: int = 30,
                            font_scale: float = 0.8,
                            color: tuple = (0, 0, 0),
                            outline_color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Draw readable text lines on an existing image.

    Text is drawn twice:
      1. thick outline
      2. thin foreground

    Parameters
    ----------
    image : np.ndarray
        Existing BGR image.
    text_lines : list[str]
    origin : tuple
        (x, y) of the first text line.
    line_spacing : int
        Vertical spacing between lines.
    font_scale : float
    color : tuple
        Foreground BGR color.
    outline_color : tuple
        Outline BGR color.

    Returns
    -------
    image : np.ndarray
    """
    x0, y0 = origin

    for i, text in enumerate(text_lines):
        y = y0 + i * line_spacing

        cv2.putText(
            image,
            str(text),
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            outline_color,
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            str(text),
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    return image