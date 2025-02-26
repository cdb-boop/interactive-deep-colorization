from __future__ import print_function

import numpy as np
import numpy.typing as npt
import cv2
import os
import pickle
from typing import Any
import warnings
from PyQt5.QtGui import QColor


def debug_trace() -> None:
    from PyQt5.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def PickleLoad(file_name: str) -> Any:
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(file_name, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    return data


def PickleSave(file_name: str, data: Any) -> None:
    with open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def print_numpy(x: Any, val: bool = True, shp: bool = False) -> None:
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def CVShow(im: np.ndarray, im_name: str = '', wait: int = 1) -> Any:
    if len(im.shape) >= 3 and im.shape[2] == 3:
        im_show = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im_show = im

    cv2.imshow(im_name, im_show)
    cv2.waitKey(wait)
    return im_show


def average_image(imgs: Any, weights: Any) -> Any:
    im_weights = np.tile(weights[:, np.newaxis, np.newaxis, np.newaxis], (1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    imgs_f = imgs.astype(np.float32)
    weights_norm = np.mean(im_weights)
    average_f = np.mean(imgs_f * im_weights, axis=0) / weights_norm
    average = average_f.astype(np.uint8)
    return average


def mkdirs(paths: list[str]) -> None:
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        warnings.warn(f"Unable to make directory: {path}.", RuntimeWarning)


def grid_vis(X: Any, nh: int, nw: int) -> Any:  # [buggy]
    if X.shape[0] == 1:
        return X[0]

    # nc = 3
    if X.ndim == 3:
        X = X[..., np.newaxis]
    if X.shape[-1] == 1:
        X = np.tile(X, [1, 1, 1, 3])

    h, w = X[0].shape[:2]

    if X.dtype == np.uint8:
        img = np.ones((h * nh, w * nw, 3), np.uint8) * 255
    else:
        img = np.ones((h * nh, w * nw, 3), X.dtype)

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    img = np.squeeze(img)
    return img

def ndarray_to_qcolor(a: npt.NDArray[np.uint8]) -> QColor:
    return QColor(a[0], a[1], a[2])

def qcolor_to_ndarray(c: QColor) -> npt.NDArray[np.uint8]:
    return np.array((c.red(), c.green(), c.blue())).astype('uint8')