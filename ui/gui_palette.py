from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QPaintEvent, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
import numpy as np
import numpy.typing as npt


class GUIPalette(QWidget):
    color_selected = pyqtSignal(np.ndarray)  # 1x3 uint8

    def __init__(self, grid_sz: tuple[int, int] = (6, 3)):
        QWidget.__init__(self)
        self.color_width = 25
        self.border = 6
        self.win_width = grid_sz[0] * self.color_width + (grid_sz[0] + 1) * self.border
        self.win_height = grid_sz[1] * self.color_width + (grid_sz[1] + 1) * self.border
        self.setFixedSize(self.win_width, self.win_height)
        self.num_colors = grid_sz[0] * grid_sz[1]
        self.grid_sz = grid_sz
        self.colors = np.array([], np.uint8)
        self.color_id = -1
        self.reset()

    def set_colors(self, colors: npt.NDArray[np.float32]) -> None:
        if colors.shape[0] == 0:
            self.colors = np.array([], np.uint8)
        else:
            self.colors = (colors[:min(colors.shape[0], self.num_colors), :] * 255).astype(np.uint8)
        self.color_id = -1
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), Qt.GlobalColor.white)
        for n, c in enumerate(self.colors):
            ca = QColor(c[0], c[1], c[2], 255)
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.setBrush(ca)
            grid_x = n % self.grid_sz[0]
            grid_y = (n - grid_x) // self.grid_sz[0]
            x = grid_x * (self.color_width + self.border) + self.border
            y = grid_y * (self.color_width + self.border) + self.border

            if n == self.color_id:
                painter.drawEllipse(x, y, self.color_width, self.color_width)
            else:
                painter.drawRoundedRect(x, y, self.color_width, self.color_width, 2, 2)

        painter.end()

    def sizeHint(self) -> QSize:
        return QSize(self.win_width, self.win_height)

    def reset(self) -> None:
        self.colors = np.array([], np.uint8)
        self.mouseClicked = False
        self.color_id = -1
        self.update()

    def selected_color(self, pos: QPoint) -> int:
        width = self.color_width + self.border
        dx = pos.x() % width
        dy = pos.y() % width
        if dx >= self.border and dy >= self.border:
            x_id = (pos.x() - dx) // width
            y_id = (pos.y() - dy) // width
            color_id = x_id + y_id * self.grid_sz[0]
            return int(color_id)
        else:
            return -1

    def update_ui(self, color_id: int) -> None:
        assert isinstance(color_id, int), "GUIPalette: Expected 'color_id' of type 'int'"
        assert self.colors is not None, "GUIPalette: colors cannot be 'None'"
        self.color_id = color_id
        self.update()
        if color_id >= 0 and color_id < len(self.colors):
            print(f"GUIPalette: Selected palette index {color_id}")
            color = self.colors[color_id]
            self.color_selected.emit(color)
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        assert self.colors is not None, "GUIPalette: colors cannot be 'None'"
        if event.button() == Qt.MouseButton.LeftButton:
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)
            self.mouseClicked = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mouseClicked:
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mouseClicked = False
