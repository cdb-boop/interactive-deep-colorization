import numpy as np
import importlib
skcolor = importlib.import_module("skimage.color")  # ignore import issue with skimage
from PyQt5.QtGui import QColor
import warnings
from enum import Enum


class ColorFormat(Enum):
    RGB = 1
    LAB = 2


def qcolor2lab_1d(qc: QColor) -> np.ndarray:
    # take QColor and do color conversion
    c = np.array([qc.red(), qc.green(), qc.blue()], np.uint8)
    return rgb2lab_1d(c)


def rgb2lab_1d(in_rgb: np.ndarray) -> np.ndarray:
    # take 1d numpy array and do color conversion
    # print('in_rgb', in_rgb)
    return skcolor.rgb2lab(in_rgb[np.newaxis, np.newaxis, :]).flatten()


def lab2rgb_1d(in_lab, clip: bool = True, dtype: str = 'uint8') -> np.ndarray:
    warnings.filterwarnings("ignore")
    tmp_rgb = skcolor.lab2rgb(in_lab[np.newaxis, np.newaxis, :]).flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    if dtype == 'uint8':
        tmp_rgb = np.round(tmp_rgb * 255).astype('uint8')
    return tmp_rgb


def snap_ab(input_l: np.float64, input_rgb: np.ndarray, return_type: ColorFormat = ColorFormat.RGB) -> np.ndarray:
    ''' given an input lightness and rgb, snap the color into a region where l,a,b is in-gamut
    '''
    T = 20
    warnings.filterwarnings("ignore")
    input_lab = rgb2lab_1d(np.array(input_rgb))  # convert input to lab
    conv_lab = input_lab.copy()  # keep ab from input
    for t in range(T):
        conv_lab[0] = input_l  # overwrite input l with input ab
        old_lab = conv_lab
        tmp_rgb = skcolor.lab2rgb(conv_lab[np.newaxis, np.newaxis, :]).flatten()
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
        conv_lab = skcolor.rgb2lab(tmp_rgb[np.newaxis, np.newaxis, :]).flatten()
        dif_lab = np.sum(np.abs(conv_lab - old_lab))
        if dif_lab < 1:
            break
        # print(conv_lab)

    conv_rgb_ingamut = lab2rgb_1d(conv_lab, clip=True, dtype='uint8')
    if (return_type == ColorFormat.RGB):
        return conv_rgb_ingamut
    elif(return_type == ColorFormat.LAB):
        conv_lab_ingamut = rgb2lab_1d(conv_rgb_ingamut)
        return conv_lab_ingamut
    else:
        raise Exception("Color format unknown")


class abGrid():
    def __init__(self, gamut_size: int = 110, D: int = 1):
        self.D = D
        self.vals_b, self.vals_a = np.meshgrid(np.arange(-gamut_size, gamut_size + D, D),
                                               np.arange(-gamut_size, gamut_size + D, D))
        self.pts_full_grid = np.concatenate((self.vals_a[:, :, np.newaxis], self.vals_b[:, :, np.newaxis]), axis=2)
        self.A = self.pts_full_grid.shape[0]
        self.B = self.pts_full_grid.shape[1]
        self.AB = self.A * self.B
        self.gamut_size = gamut_size

    def update_gamut(self, l_in: np.float64) -> tuple[np.ndarray, np.ndarray]:
        warnings.filterwarnings("ignore")
        thresh = 1.0
        pts_lab = np.concatenate((l_in + np.zeros((self.A, self.B, 1)), self.pts_full_grid), axis=2)
        self.pts_rgb = (255 * np.clip(skcolor.lab2rgb(pts_lab), 0, 1)).astype('uint8')
        pts_lab_back = skcolor.rgb2lab(self.pts_rgb)
        pts_lab_diff = np.linalg.norm(pts_lab - pts_lab_back, axis=2)

        self.mask = pts_lab_diff < thresh
        mask3 = np.tile(self.mask[..., np.newaxis], [1, 1, 3])
        self.masked_rgb = self.pts_rgb.copy()
        self.masked_rgb[np.invert(mask3)] = 255
        return self.masked_rgb, self.mask

    def ab2xy(self, a: np.float64, b: np.float64) -> tuple[np.float64, np.float64]:
        y = self.gamut_size + a
        x = self.gamut_size + b
        # print('ab2xy (%d, %d) -> (%d, %d)' % (a, b, x, y))
        return x, y

    def xy2ab(self, x: int, y: int) -> tuple[np.float64, np.float64]:
        a = np.float64(y) - self.gamut_size
        b = np.float64(x) - self.gamut_size
        # print('xy2ab (%d, %d) -> (%d, %d)' % (x, y, a, b))
        return a, b
