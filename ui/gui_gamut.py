import cv2
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QImage, QPaintEvent, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QPoint, QSize
from data import lab_gamut
import numpy as np
import numpy.typing as npt
import warnings


class GUIGamut(QWidget):
    color_selected = pyqtSignal(np.ndarray)  # 1x3 uint8

    def __init__(self, gamut_size: int = 110):
        QWidget.__init__(self)
        self.gamut_size = gamut_size
        self.win_size = gamut_size * 2  # divided by 4
        self.setFixedSize(self.win_size, self.win_size)
        self.ab_grid = lab_gamut.abGrid(gamut_size=gamut_size, D=1)
        self.reset()
        self.mask: None | npt.NDArray[np.bool_] = None
        self.pos: None | QPoint = None

    def set_gamut(self, l_in: np.float64 = np.float64(50)) -> None:
        self.l_in = l_in
        self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=l_in)
        self.update()

    def set_ab(self, color: np.ndarray) -> None:
        assert color.shape == (3, ), "GUIGamut.set_ab(...) expects color of shape(3,)"
        self.color = color
        self.lab = lab_gamut.rgb2lab_1d(self.color)
        x, y = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        h = self.win_size // 2
        x, y = int(x - h) + h, int(y - h) + h  # round towards center
        self.pos = QPoint(x, y)
        self.update()

    def is_valid_point(self, pos: QPoint) -> bool:
        if not isinstance(pos, QPoint):
            warnings.warn(f"GUIGamut: 'pos' was not 'QPoint'", RuntimeWarning)
            return False
        if self.mask is None:
            warnings.warn(f"GUIGamut: 'mask' was 'None'", RuntimeWarning)
            return False

        x = pos.x()
        y = pos.y()
        if x >= 0 and y >= 0 and x < self.win_size and y < self.win_size:
            return self.mask[y, x].astype(bool)
        else:
            return False

    def update_ui(self, pos: QPoint) -> None:
        self.pos = pos
        a, b = self.ab_grid.xy2ab(pos.x(), pos.y())
        # get color we need L
        L = self.l_in
        lab = np.array([L, a, b])
        color = lab_gamut.lab2rgb_1d(lab, clip=True, dtype='uint8')
        self.color_selected.emit(color)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), Qt.GlobalColor.white)
        if self.ab_map is not None:
            ab_map = cv2.resize(self.ab_map, (self.win_size, self.win_size))
            qImg = QImage(ab_map.tostring(), self.win_size, self.win_size, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        painter.setPen(QPen(Qt.GlobalColor.gray, 3, Qt.PenStyle.DotLine, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(self.win_size // 2, 0, self.win_size // 2, self.win_size)
        painter.drawLine(0, self.win_size // 2, self.win_size, self.win_size // 2)
        if self.pos is not None:
            painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
            w = 5
            x = self.pos.x()
            y = self.pos.y()
            painter.drawLine(QPointF(x - w, y), QPointF(x + w, y))
            painter.drawLine(QPointF(x, y - w), QPointF(x, y + w))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton and self.is_valid_point(pos):  # click the point
            self.update_ui(pos)
            self.mouseClicked = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        if self.is_valid_point(pos):
            if self.mouseClicked:
                self.update_ui(pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mouseClicked = False

    def sizeHint(self) -> QSize:
        return QSize(self.win_size, self.win_size)

    def reset(self) -> None:
        self.ab_map = None
        self.mask = None
        self.color = None
        self.lab = None
        self.pos = None
        self.mouseClicked = False
        self.update()
