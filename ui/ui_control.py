import numpy as np
import numpy.typing as npt
from PyQt5.QtGui import QColor, QPen, QPainter
from PyQt5.QtCore import Qt, QRectF, QPoint
import cv2
import warnings


class UserEdit(object):
    def __init__(self, mode, win_size: int, load_size: int, img_size: tuple[int, int]):
        self.mode = mode
        self.win_size = win_size
        self.img_size = img_size
        self.load_size = load_size
        print(f"UserEdit: Image size {self.img_size}")
        max_width = np.max(self.img_size)
        self.scale = float(max_width) / self.load_size
        self.dw = int((self.win_size - img_size[0]) // 2)
        self.dh = int((self.win_size - img_size[1]) // 2)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.ui_count = 0
        print(self)

    def scale_point(self, in_x: int, in_y: int, w: int) -> tuple[int, int]:
        x = int((in_x - self.dw) / float(self.img_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.img_h) * self.load_size) + w
        return x, y

    def __str__(self):
        return f"UserEdit: Add [{self.mode}] with win_size {self.win_size:3.3f}, load_size {self.load_size:3.3f}."


class PointEdit(UserEdit):
    def __init__(self, win_size: int, load_size: int, img_size: tuple[int, int]):
        UserEdit.__init__(self, 'point', win_size, load_size, img_size)

    def add(self, pnt: QPoint, color: QColor, userColor: QColor, width: float, ui_count: int):
        self.pnt = pnt
        self.color = color
        self.userColor = userColor
        self.width = width
        self.ui_count = ui_count

    def select_old(self, pnt: QPoint, ui_count: int) -> tuple[QColor, float]:
        self.pnt = pnt
        self.ui_count = ui_count
        return self.userColor, self.width

    def update_color(self, color: QColor, userColor: QColor) -> None:
        self.color = color
        self.userColor = userColor

    def updateInput(self, im: np.ndarray, mask: np.ndarray, vis_im: np.ndarray) -> None:
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.scale_point(pnt.x(), pnt.y(), -w)
        tl = (x1, y1)
        x2, y2 = self.scale_point(pnt.x(), pnt.y(), w)
        br = (x2, y2)
        c = (self.color.red(), self.color.green(), self.color.blue())
        uc = (self.userColor.red(), self.userColor.green(), self.userColor.blue())
        cv2.rectangle(mask, tl, br, 255, -1)
        cv2.rectangle(im, tl, br, c, -1)
        cv2.rectangle(vis_im, tl, br, uc, -1)

    def is_same(self, pnt: QPoint) -> bool:
        dx = abs(self.pnt.x() - pnt.x())
        dy = abs(self.pnt.y() - pnt.y())
        return dx <= self.width + 1 and dy <= self.width + 1

    def update_painter(self, painter: QPainter) -> None:
        w = max(3, self.width)
        c = self.color
        r = c.red()
        g = c.green()
        b = c.blue()
        ca = QColor(c.red(), c.green(), c.blue(), 255)
        d_to_black = r * r + g * g + b * b
        d_to_white = (255 - r) * (255 - r) + (255 - g) * (255 - g) + (255 - r) * (255 - r)
        if d_to_black > d_to_white:
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
        else:
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(ca)
        painter.drawRoundedRect(QRectF(self.pnt.x() - w, self.pnt.y() - w, 1 + 2 * w, 1 + 2 * w), 2, 2)


class UIControl:
    def __init__(self, win_size: int = 256, load_size: int = 512):
        self.win_size = win_size
        self.load_size = load_size
        self.reset()
        self.userEdit: None | PointEdit = None
        self.userEdits: list[PointEdit] = []
        self.ui_count: int = 0

    def setImageSize(self, img_size: tuple[int, int]) -> None:
        self.img_size = img_size

    def addStroke(self, prevPnt: QPoint, nextPnt: QPoint, color: QColor, userColor: QColor, width: float) -> None:
        warnings.warn("UIControl: 'addStroke()' unimplemented", RuntimeWarning)
        pass

    def erasePoint(self, pnt: QPoint) -> bool:
        isErase = False
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdits.remove(ue)
                print(f"UIControl: Removed user edit {id}\n")
                isErase = True
                break
        return isErase

    def addPoint(self, pnt: QPoint, color: QColor, userColor: QColor, width: float) -> tuple[QColor, float, bool]:
        self.ui_count += 1
        self.userEdit = None
        isNew = True
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdit = ue
                isNew = False
                print(f"UIControl: Selected user edit {id}.\n")
                break

        if self.userEdit is None:
            self.userEdit = PointEdit(self.win_size, self.load_size, self.img_size)
            print(f"UIControl: Added user edit at index {len(self.userEdits)}.\n")
            self.userEdits.append(self.userEdit)
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
            return userColor, width, isNew
        else:
            userColor, width = self.userEdit.select_old(pnt, self.ui_count)
            return userColor, width, isNew

    def movePoint(self, pnt: QPoint, color: QColor, userColor: QColor, width: float) -> None:
        if self.userEdit is not None:
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
        else:
            warnings.warn("UIControl: Cannot move user edit 'None'", RuntimeWarning)

    def update_color(self, color: QColor, userColor: QColor) -> None:
        if self.userEdit is not None:
            self.userEdit.update_color(color, userColor)
        else:
            warnings.warn("UIControl: Cannot update color of user edit 'None'", RuntimeWarning)

    def update_painter(self, painter: QPainter) -> None:
        for ue in self.userEdits:
            if ue is not None:
                ue.update_painter(painter)

    def get_stroke_image(self, im: np.ndarray) -> np.ndarray:
        warnings.warn("UIControl: 'get_stroke_image()' unimplemented.", RuntimeWarning)
        return im

    def get_recently_used_colors(self) -> npt.NDArray[np.float32]:
        if len(self.userEdits) == 0:
            return np.array([], np.uint8)
        nEdits = len(self.userEdits)
        ui_counts = np.zeros(nEdits)
        ui_colors = np.zeros((nEdits, 3))
        for n, ue in enumerate(self.userEdits):
            ui_counts[n] = ue.ui_count
            c = ue.userColor
            ui_colors[n, :] = [c.red(), c.green(), c.blue()]

        ui_counts = np.array(ui_counts)
        ids = np.argsort(-ui_counts)
        ui_colors = ui_colors[ids, :]
        unique_colors = []
        for ui_color in ui_colors:
            is_exit = False
            for u_color in unique_colors:
                d = np.sum(np.abs(u_color - ui_color))
                if d < 0.1:
                    is_exit = True
                    break

            if not is_exit:
                unique_colors.append(ui_color)

        unique_colors = np.vstack(unique_colors)
        return unique_colors / 255.0

    def get_input(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        vis_im = np.zeros((h, w, 3), np.uint8)

        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)

        return im, mask

    def reset(self) -> None:
        self.userEdits = []
        self.userEdit = None
        self.ui_count = 0
