import cv2
import numpy as np


def load_video(path: str):
    """
    Charge une vidéo et retourne les frames en niveaux de gris.

    Retourne
    --------
    frames : list of np.ndarray
    fps    : float
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    print(f"Chargé : {len(frames)} frames à {fps:.1f} fps")
    return frames, fps


def enhance_contrast(frame: np.ndarray,
                     clip_limit: float = 2.0,
                     tile_size: int = 8) -> np.ndarray:
    """
    CLAHE — amélioration locale du contraste.
    Indispensable avant Canny et Hough sur images écho.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    return clahe.apply(frame)


def gaussian_blur(frame: np.ndarray,
                  kernel_size: int = 5) -> np.ndarray:
    """
    Lissage gaussien pour atténuer le bruit de speckle.
    kernel_size doit être impair.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def preprocess(frame: np.ndarray,
               contrast: bool = True,
               blur: bool = True,
               blur_kernel: int = 5) -> np.ndarray:
    """
    Pipeline complet de prétraitement pour une frame.
    Ordre : CLAHE → Gaussian blur.
    """
    if contrast:
        frame = enhance_contrast(frame)
    if blur:
        frame = gaussian_blur(frame, kernel_size=blur_kernel)
    return frame


def crop_roi(frame: np.ndarray,
             x: int, y: int,
             w: int, h: int) -> np.ndarray:
    """
    Découpe une région d'intérêt (ROI) dans la frame.
    Utile pour se concentrer sur la zone musculaire.
    """
    return frame[y:y+h, x:x+w]
