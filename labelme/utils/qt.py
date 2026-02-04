import os.path as osp
from math import sqrt

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

here = osp.dirname(osp.abspath(__file__))


def newIcon(icon_file_name: str, disabled_opacity: float = 0.3) -> QtGui.QIcon:
    if osp.splitext(icon_file_name)[1] == "":
        icon_file_name = f"{icon_file_name}.png"  # XXX: convention
    icons_dir: str = osp.join(here, "../icons")
    icon_path = osp.join(":/", icons_dir, icon_file_name)
    icon = QtGui.QIcon(icon_path)

    # Add disabled (grayed out) mode for the icon
    sizes = [16, 24, 32, 48, 64]
    for size in sizes:
        pixmap = icon.pixmap(size, size, QtGui.QIcon.Normal, QtGui.QIcon.Off)
        if not pixmap.isNull():
            # Create grayed out version
            disabled_pixmap = QtGui.QPixmap(pixmap.size())
            disabled_pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(disabled_pixmap)
            painter.setOpacity(disabled_opacity)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()
            icon.addPixmap(disabled_pixmap, QtGui.QIcon.Disabled, QtGui.QIcon.Off)

    return icon


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
    disabled_opacity: float = 0.3,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(" ", "\n"))
        a.setIcon(newIcon(icon, disabled_opacity=disabled_opacity))
    if shortcut is not None:
        if isinstance(shortcut, list | tuple):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r"^[^ \t].+"), None)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return np.linalg.norm(p3 - p1)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fmtShortcut(text):
    mod, key = text.split("+", 1)
    return f"<b>{mod}</b>+<b>{key}</b>"
