from __future__ import annotations

import base64
import csv
import datetime
import enum
import functools
import html
import json
import math
import os
import os.path as osp
import platform
import re
import subprocess
import types
import webbrowser
from pathlib import Path
from typing import Literal

import locale

import imgviz
import numpy as np
import osam
from loguru import logger
from numpy.typing import NDArray
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from labelme import __appname__
from labelme import __version__
from labelme._automation import bbox_from_text
from labelme._automation._osam_session import OsamSession
from labelme._label_file import LabelFile
from labelme._label_file import LabelFileError
from labelme._label_file import ShapeDict
from labelme.config import load_config
from labelme.shape import Shape
from labelme.widgets import AiAssistedAnnotationWidget
from labelme.widgets import AiTextToAnnotationWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import NavigatorWidget
from labelme.widgets import StatusStats
from labelme.widgets import UpdateDistributionWidget
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme.widgets import download_ai_model

from . import utils

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".

# handle high-dpi scaling issue
# https://leomoon.com/journal/python/high-dpi-scaling-in-pyqt5
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


# Use matplotlib's tab20 colormap for label colors
# Exclude gray (14,15) for better visibility
# Reorder: dark colors first, then light colors
import matplotlib.pyplot as plt
_tab20_cmap = plt.cm.get_cmap("tab20")
# Keep: blue(0,1), orange(2,3), green(4,5), red(6,7), purple(8,9), brown(10,11), pink(12,13), olive(16,17), cyan(18,19)
_keep_dark = [0, 2, 4, 6, 8, 10, 12, 16, 18]
_keep_light = [1, 3, 5, 7, 9, 11, 13, 17, 19]
_dark_colors = [_tab20_cmap(i / 20)[:3] for i in _keep_dark]
_light_colors = [_tab20_cmap(i / 20)[:3] for i in _keep_light]
LABEL_COLORMAP: NDArray[np.uint8] = np.array(
    _dark_colors + _light_colors, dtype=np.float32
) * 255
LABEL_COLORMAP = LABEL_COLORMAP.astype(np.uint8)
# Replace dark red with custom red #b7282e
LABEL_COLORMAP[3] = [183, 40, 46]  # dark red position (index 6 in tab20 -> position 3 in dark colors)
# Replace dark brown with custom color #556B2F
LABEL_COLORMAP[5] = [85, 107, 47]  # dark brown position (index 10 in tab20 -> position 5 in dark colors)


class _ZoomMode(enum.Enum):
    FIT_WINDOW = enum.auto()
    FIT_WIDTH = enum.auto()
    MANUAL_ZOOM = enum.auto()


_AI_TEXT_TO_ANNOTATION_CREATE_MODE_TO_SHAPE_TYPE: dict[
    str, Literal["mask", "polygon", "rectangle"]
] = {
    "ai_mask": "mask",
    "ai_polygon": "polygon",
    "polygon": "polygon",
    "rectangle": "rectangle",
}


class _FileListItemDelegate(QtWidgets.QStyledItemDelegate):
    """Render selection as a border overlay so the item's own background
    (the mtime-gradient color set via setBackground) remains visible."""

    BORDER_COLOR = QtGui.QColor(13, 71, 161)
    BORDER_WIDTH = 2

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        is_selected = bool(opt.state & QtWidgets.QStyle.State_Selected)
        # Strip the Selected state so super paints the item's own background
        # (no gray/blue system selection overlay).
        opt.state &= ~QtWidgets.QStyle.State_Selected
        super().paint(painter, opt, index)
        if is_selected:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
            pen = QtGui.QPen(self.BORDER_COLOR, self.BORDER_WIDTH)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            inset = self.BORDER_WIDTH / 2
            rect = QtCore.QRectF(option.rect).adjusted(inset, inset, -inset, -inset)
            painter.drawRect(rect)
            painter.restore()


class MainWindow(QtWidgets.QMainWindow):
    _config_file: Path | None
    _config: dict
    # True while we mirror a canvas selection into the label list; the class
    # default keeps the guard valid before __init__ sets it.
    _syncing_label_selection = False
    _dock_refresh_pending = False

    # File list background color: blue, alpha gradient by mtime (oldest=20, newest=70)
    FILE_COLOR_RGB = (30, 136, 229)
    FILE_ALPHA_MIN = 20  # oldest mtime (薄い青)
    FILE_ALPHA_MAX = 70  # newest mtime (濃い青)
    FILE_ANNOTATED_COLOR = QtGui.QColor(30, 136, 229, FILE_ALPHA_MIN)  # legacy fallback (alpha 20)
    FILE_NEWLY_SAVED_COLOR = QtGui.QColor(30, 136, 229, FILE_ALPHA_MAX)  # legacy fallback (alpha 70)
    # Red background for shapes without modification timestamp
    SHAPE_UNMODIFIED_COLOR = QtGui.QColor(229, 57, 53, 20)  # rgba with low alpha (matching file list style)
    # Predefined zoom levels (in percent)
    ZOOM_LEVELS = (25, 33, 50, 67, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1150, 1300, 1450, 1600, 1800, 2000)

    filename: str | None
    _text_osam_session: OsamSession | None = None
    _is_changed: bool = False
    _copied_shapes: list[Shape]
    _zoom_mode: _ZoomMode
    _zoom_values: dict[str, tuple[_ZoomMode, int]]
    _brightness_contrast_values: dict[str, tuple[int | None, int | None]]
    _prev_opened_dir: str | None
    _initially_annotated_files: set[str]  # Files already annotated when dir was opened
    _current_file_row: int  # Row index of the currently loaded file in the file list
    _other_data: dict | None
    # mtime cache for gradient coloring of file list (path → mtime)
    _file_mtimes: dict[str, float]
    _file_mtime_range: tuple[float, float] | None

    # NB: this tells Mypy etc. that `actions` here
    #     is a different type cf. the parent class
    #     (where it is Callable[[QWidget], list[QAction]]).
    actions: types.SimpleNamespace  # type: ignore[assignment]

    def __init__(
        self,
        config_file: Path | None = None,
        config_overrides: dict | None = None,
        filename: str | None = None,
        output: str | None = None,
        output_file: str | None = None,
        output_dir: str | None = None,
        prompt_for_dir: bool = False,
    ) -> None:
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output
        del output

        super().__init__()
        self.setWindowTitle(__appname__)

        self._config_file, self._config = self._load_config(
            config_file=config_file, config_overrides=config_overrides
        )

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        self._copied_shapes = []
        self._last_label = None  # Last used label (for changeSame action)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.labelList.setStyleSheet("QListView::item { min-height: 24px; padding: 2px 0px; }")
        self._prev_opened_dir = None
        self._initially_annotated_files: set[str] = set()
        self._current_file_row: int = -1
        self._file_mtimes = {}
        self._file_mtime_range = None

        # Navigator (minimap)
        self.navigator, self.navigator_dock = self._create_navigator_dock()
        nav_features = (
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetMovable
        )
        self.navigator_dock.setFeatures(nav_features)

        # Update Distribution (heatmap of recent modifications)
        self.update_distribution, self.update_distribution_dock = (
            self._create_update_distribution_dock()
        )
        self.update_distribution_dock.setFeatures(nav_features)

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if self._config["flags"]:
            self._load_flags(flags={k: False for k in self._config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        # Set while we push a selection into the label list ourselves, so the
        # handler below ignores the echo (see shapeSelectionChanged).
        self._syncing_label_selection = False
        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setColormap(LABEL_COLORMAP)
        self.uniqLabelList.setToolTip(
            self.tr("Select label to start annotating for it. Press 'Esc' to deselect.")
        )
        if self._config["labels"]:
            for label in sorted(self._config["labels"]):
                self.uniqLabelList.add_label_item(
                    label=label, color=self._get_rgb_by_label(label=label),
                    sorted_insert=True,
                )
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.setStyleSheet("QListWidget::item { min-height: 24px; padding: -3px; }")
        self.fileListWidget.setItemDelegate(_FileListItemDelegate(self.fileListWidget))
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()

        # Line opacity spinbox
        self.lineOpacityWidget = QtWidgets.QSpinBox()
        self.lineOpacityWidget.setRange(0, 95)  # Max 95% to keep lines visible
        self.lineOpacityWidget.setSuffix(" %")
        self.lineOpacityWidget.setSingleStep(5)  # 5% increments
        self.lineOpacityWidget.valueChanged.connect(self._line_opacity_changed)
        self.lineOpacityWidget.setValue(70)  # Default 70% transparency

        # Vertex opacity spinbox (for polygon corners)
        self.pointOpacityWidget = QtWidgets.QSpinBox()
        self.pointOpacityWidget.setRange(0, 95)  # Max 95% to keep vertices visible
        self.pointOpacityWidget.setSuffix(" %")
        self.pointOpacityWidget.setSingleStep(5)  # 5% increments
        self.pointOpacityWidget.valueChanged.connect(self._vertex_opacity_changed)
        self.pointOpacityWidget.setValue(50)  # Default 50% transparency

        # Fill opacity spinbox
        self.fillOpacityWidget = QtWidgets.QSpinBox()
        self.fillOpacityWidget.setRange(0, 100)
        self.fillOpacityWidget.setSuffix(" %")
        self.fillOpacityWidget.setSingleStep(5)  # 5% increments
        self.fillOpacityWidget.valueChanged.connect(self._fill_opacity_changed)
        self.fillOpacityWidget.setValue(80)  # Default 80% transparency

        # Line width spinbox
        self.lineWidthWidget = QtWidgets.QSpinBox()
        self.lineWidthWidget.setRange(1, 10)
        self.lineWidthWidget.setSuffix(" px")
        self.lineWidthWidget.valueChanged.connect(self._line_width_changed)
        self.lineWidthWidget.setValue(6)  # Default line width

        # Point object size spinbox (applies only to point shape_type)
        self.pointObjectSizeWidget = QtWidgets.QSpinBox()
        self.pointObjectSizeWidget.setRange(4, 96)
        self.pointObjectSizeWidget.setSuffix(" px")
        self.pointObjectSizeWidget.valueChanged.connect(
            self._point_object_size_changed
        )
        self.pointObjectSizeWidget.setValue(24)

        # Vertex size spinbox (applies to polygon/rectangle vertices)
        self.vertexSizeWidget = QtWidgets.QSpinBox()
        self.vertexSizeWidget.setRange(2, 32)
        self.vertexSizeWidget.setSuffix(" px")
        self.vertexSizeWidget.valueChanged.connect(self._vertex_size_changed)
        self.vertexSizeWidget.setValue(8)

        self.storeImageDataCheckbox = QtWidgets.QCheckBox()
        self.storeImageDataCheckbox.toggled.connect(self._store_image_data_toggled)

        self.customCursorCheckbox = QtWidgets.QCheckBox()
        self.customCursorCheckbox.toggled.connect(self._custom_cursor_toggled)

        self.rightClickEditCheckbox = QtWidgets.QCheckBox()
        self.rightClickEditCheckbox.toggled.connect(self._right_click_edit_toggled)

        self.skipDeleteConfirmCheckbox = QtWidgets.QCheckBox()

        self.skipSaveNameConfirmCheckbox = QtWidgets.QCheckBox()

        self.parallelLineDistCheckbox = QtWidgets.QCheckBox()
        self.parallelLineDistCheckbox.toggled.connect(
            self._parallel_line_dist_toggled
        )

        self.textBoundingCheckbox = QtWidgets.QCheckBox()
        self.textBoundingCheckbox.toggled.connect(
            self._text_bounding_toggled
        )

        self.lineFitCheckbox = QtWidgets.QCheckBox()
        self.lineFitCheckbox.toggled.connect(
            self._line_fit_toggled
        )

        self.darkPixelMagnetCheckbox = QtWidgets.QCheckBox()
        self.darkPixelMagnetCheckbox.toggled.connect(
            self._dark_pixel_magnet_toggled
        )

        self.autoFitCheckbox = QtWidgets.QCheckBox()
        self.autoFitCheckbox.toggled.connect(
            self._auto_fit_toggled
        )

        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
            auto_fit_tolerance=self._config["canvas"].get("auto_fit_tolerance", 0.2),
        )
        self.canvas.zoomRequest.connect(self._zoom_requested)
        self.canvas.pinchZoomRequest.connect(self._pinch_zoom_requested)
        self.canvas.mouseMoved.connect(self._update_status_stats)
        self.canvas.statusUpdated.connect(lambda text: self.status_left.setText(text))
        self.canvas.setParallelLineMagnetConfig(
            self._magnet_rules("parallel_line_magnet")
        )
        self.canvas.setTextBoundingMagnetConfig(
            self._magnet_rules("text_bounding_magnet")
        )
        self.canvas.setLineFitMagnetConfig(self._magnet_rules("line_fit_magnet"))
        self.canvas.setDarkPixelMagnetConfig(
            self._magnet_rules("dark_pixel_magnet")
        )
        self.uniqLabelList.itemSelectionChanged.connect(
            self._update_pending_draw_label
        )

        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        # Make scroll bars thicker and arrow buttons wider
        self.scrollArea.setStyleSheet("""
            QScrollBar:vertical {
                width: 20px;
            }
            QScrollBar:horizontal {
                height: 20px;
            }
            QScrollBar::sub-line:vertical,
            QScrollBar::add-line:vertical {
                height: 28px;
            }
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-line:horizontal {
                width: 28px;
            }
        """)
        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        # Connect scrollbar value changes to navigator update
        self.scrollBars[Qt.Vertical].valueChanged.connect(self._updateNavigatorViewport)
        self.scrollBars[Qt.Horizontal].valueChanged.connect(self._updateNavigatorViewport)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.shapeMoved.connect(self._recompute_reference_medians)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.editModeChanged.connect(self._onEditModeChanged)

        self.setCentralWidget(self.scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.setDockNestingEnabled(True)
        self.setDockOptions(
            self.dockOptions() | QtWidgets.QMainWindow.AllowNestedDocks
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.navigator_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.update_distribution_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock, Qt.Vertical)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            icon=None,
            tip=self.tr("Quit application"),
        )
        open_config = action(
            text=self.tr("Preferences…"),
            slot=self._open_config_file,
            shortcut="Ctrl+," if platform.system() == "Darwin" else "Ctrl+Shift+,",
            icon=None,
            tip=self.tr("Open config file in text editor"),
        )
        open_config.setMenuRole(QtWidgets.QAction.PreferencesRole)
        open_ = action(
            self.tr("&Open\n"),
            self._open_file_with_dialog,
            shortcuts["open"],
            icon="folder-open.svg",
            tip=self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self._open_dir_with_dialog,
            shortcuts["open_dir"],
            icon="folder-open.svg",
            tip=self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self._open_next_image,
            shortcuts["open_next"],
            icon="arrow-fat-right.svg",
            tip=self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self._open_prev_image,
            shortcuts["open_prev"],
            icon="arrow-fat-left.svg",
            tip=self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            icon="floppy-disk.svg",
            tip=self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            icon="floppy-disk.svg",
            tip=self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            icon="file-x.svg",
            tip=self.tr("Delete current label file"),
            enabled=False,
        )

        exportFileList = action(
            self.tr("Export\n&Report"),
            self.exportFileList,
            None,
            icon="table-export.svg",
            tip=self.tr("Export annotation report to CSV"),
            enabled=False,
            disabled_opacity=0.6,
        )

        progressStats = action(
            self.tr("Progress\n&Stats"),
            self.progressStats,
            None,
            icon="chart-bar.svg",
            tip=self.tr("Generate progress statistics chart"),
            enabled=False,
            disabled_opacity=0.6,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="folders.svg",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            icon="x-circle.svg",
            tip=self.tr("Close current file"),
        )

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self._switch_canvas_mode(edit=False, createMode="polygon"),
            shortcuts["create_polygon"],
            "polygon.svg",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createPolygon3Mode = action(
            self.tr("3点ポリゴン\nを作成"),
            lambda: self._switch_canvas_mode(edit=False, createMode="polygon3"),
            None,
            "polygon3.svg",
            self.tr("3点を置くとポリゴンが確定します"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self._switch_canvas_mode(edit=False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "rectangle.svg",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self._switch_canvas_mode(edit=False, createMode="circle"),
            shortcuts["create_circle"],
            "circle.svg",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createCircleMode.setVisible(False)
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self._switch_canvas_mode(edit=False, createMode="line"),
            shortcuts["create_line"],
            "line-segment.svg",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createLineMode.setVisible(False)
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self._switch_canvas_mode(edit=False, createMode="point"),
            shortcuts["create_point"],
            icon="circles-four.svg",
            tip=self.tr("Start drawing points"),
            enabled=False,
        )
        createMagicWandMode = action(
            self.tr("魔法の杖で\nポリゴン作成"),
            lambda: self._switch_canvas_mode(edit=False, createMode="magic_wand"),
            None,
            "magic-wand.svg",
            self.tr("魔法の杖でポリゴンを作成"),
            enabled=False,
        )
        createMagicWandMode.setVisible(False)
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self._switch_canvas_mode(edit=False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "line-segments.svg",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createLineStripMode.setVisible(False)
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self._switch_canvas_mode(edit=False, createMode="ai_polygon"),
            None,
            "ai-polygon.svg",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode.setVisible(False)
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self._switch_canvas_mode(edit=False, createMode="ai_mask"),
            None,
            "ai-mask.svg",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode.setVisible(False)
        editMode = action(
            self.tr("Edit Polygons"),
            lambda: self._switch_canvas_mode(edit=True),
            shortcuts["edit_polygon"],
            icon="note-pencil.svg",
            tip=self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            icon="trash.svg",
            tip=self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            icon="copy.svg",
            tip=self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        rectToTriangle = action(
            self.tr("矩形を3点\nポリゴンに変換"),
            self.convertRectanglesToTriangles,
            None,
            "rect-to-triangle.svg",
            self.tr(
                "選択中の矩形を、.labelmerc の "
                "rect_to_triangle_drop_corner で指定した隅を除いた"
                "3点ポリゴンに変換"
            ),
            enabled=False,
        )
        deleteAllShapes = action(
            self.tr("ポリゴン\n一括削除"),
            self.deleteAllShapes,
            None,
            "trash-polygon.svg",
            self.tr("開いている画像のポリゴンを全て削除"),
            enabled=False,
        )
        removeHighIouShapes = action(
            self.tr("IoU0.9超の\nポリゴン削除"),
            self.removeHighIouShapes,
            None,
            "remove-duplicate-polygon.svg",
            self.tr("IoUが0.9を超える重複ポリゴンのうち後に作られた方を削除"),
            enabled=False,
        )
        copyFromSpecifiedJson = action(
            self.tr("指定ファイルの\nポリゴン全複製"),
            self.copyShapesFromSpecifiedJson,
            None,
            "file-polygon.svg",
            self.tr("指定したJSONファイルのポリゴンを全て複製"),
            enabled=False,
        )
        copyFromPrevJson = action(
            self.tr("1つ上のファイルの\nポリゴン全複製"),
            self.copyShapesFromPreviousJson,
            None,
            "file-polygon-up.svg",
            self.tr(
                "ファイル一覧を遡り、最初に見つかる保存済みJSONの"
                "ポリゴンを全て複製"
            ),
            enabled=False,
        )
        addBase64ToAllJsons = action(
            self.tr("全JSONに\n画像データ追加"),
            self.addBase64ToAllJsons,
            None,
            "image-add-all.svg",
            self.tr(
                "ファイル一覧の全画像に対応するJSONを開き、"
                "base64の画像データを書き込む"
            ),
            enabled=False,
        )
        removeBase64FromAllJsons = action(
            self.tr("全JSONから\n画像データ削除"),
            self.removeBase64FromAllJsons,
            None,
            "image-remove-all.svg",
            self.tr(
                "ファイル一覧の全画像に対応するJSONを開き、"
                "base64の画像データを削除する"
            ),
            enabled=False,
        )
        showAutoFit = action(
            self.tr("矩形移動時に自動フィットを表示"),
            self._toggle_auto_fit_visible,
            checkable=True,
            checked=False,
        )

        showLineOpacity = action(
            self.tr("線の透明度を表示"),
            self._toggle_line_opacity_visible,
            checkable=True,
            checked=True,
        )
        showPointOpacity = action(
            self.tr("頂点の透明度を表示"),
            self._toggle_point_opacity_visible,
            checkable=True,
            checked=True,
        )
        showFillOpacity = action(
            self.tr("塗りの透明度を表示"),
            self._toggle_fill_opacity_visible,
            checkable=True,
            checked=True,
        )
        showLineWidth = action(
            self.tr("線の太さを表示"),
            self._toggle_line_width_visible,
            checkable=True,
            checked=True,
        )
        showPointObjectSize = action(
            self.tr("点の大きさを表示"),
            self._toggle_point_object_size_visible,
            checkable=True,
            checked=True,
        )
        showVertexSize = action(
            self.tr("頂点の大きさを表示"),
            self._toggle_vertex_size_visible,
            checkable=True,
            checked=True,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            icon="arrow-u-up-left.svg",
            tip=self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="trash.svg",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            icon="arrow-u-up-left.svg",
            tip=self.tr("Undo last add and edit of shape"),
            enabled=False,
        )
        redo = action(
            self.tr("やり直す"),
            self.redoShapeEdit,
            "Ctrl+Shift+Z",
            icon="arrow-u-up-right.svg",
            tip=self.tr("最後に元に戻した図形追加・編集をやり直す"),
            enabled=False,
        )
        redo.setVisible(False)

        showCreatePolygon3 = action(
            self.tr("3点ポリゴンを作成ボタンを表示"),
            self._toggle_create_polygon3_visible,
            checkable=True,
            checked=False,
        )
        showCreateCircle = action(
            self.tr("円を作成ボタンを表示"),
            self._toggle_create_circle_visible,
            checkable=True,
            checked=False,
        )
        showCreateLine = action(
            self.tr("直線を作成ボタンを表示"),
            self._toggle_create_line_visible,
            checkable=True,
            checked=False,
        )
        showCreateLineStrip = action(
            self.tr("折れ線を作成ボタンを表示"),
            self._toggle_create_linestrip_visible,
            checkable=True,
            checked=False,
        )
        showCreateAiPolygon = action(
            self.tr("AIポリゴンを作成ボタンを表示"),
            self._toggle_create_ai_polygon_visible,
            checkable=True,
            checked=False,
        )
        showCreateAiMask = action(
            self.tr("AIマスクを作成ボタンを表示"),
            self._toggle_create_ai_mask_visible,
            checkable=True,
            checked=False,
        )
        showCreateMagicWand = action(
            self.tr("魔法の杖を作成ボタンを表示"),
            self._toggle_create_magic_wand_visible,
            checkable=True,
            checked=False,
        )
        showRedo = action(
            self.tr("やり直すボタンを表示"),
            self._toggle_redo_visible,
            checkable=True,
            checked=False,
        )
        showRectToTriangle = action(
            self.tr("矩形を3点ポリゴンに変換ボタンを表示"),
            self._toggle_rect_to_triangle_visible,
            checkable=True,
            checked=True,
        )
        showDeleteAllShapes = action(
            self.tr("ポリゴン一括削除ボタンを表示"),
            self._toggle_delete_all_shapes_visible,
            checkable=True,
            checked=True,
        )
        showRemoveHighIouShapes = action(
            self.tr("IoU0.9超のポリゴン削除ボタンを表示"),
            self._toggle_remove_high_iou_visible,
            checkable=True,
            checked=True,
        )
        showCopyFromSpecifiedJson = action(
            self.tr("指定ファイルのポリゴン全複製ボタンを表示"),
            self._toggle_copy_from_specified_json_visible,
            checkable=True,
            checked=True,
        )
        showCopyFromPrevJson = action(
            self.tr("1つ上のファイルのポリゴン全複製ボタンを表示"),
            self._toggle_copy_from_prev_json_visible,
            checkable=True,
            checked=True,
        )
        showAddBase64ToAllJsons = action(
            self.tr("全JSONに画像データ追加ボタンを表示"),
            self._toggle_add_base64_to_all_jsons_visible,
            checkable=True,
            checked=False,
        )
        showRemoveBase64FromAllJsons = action(
            self.tr("全JSONから画像データ削除ボタンを表示"),
            self._toggle_remove_base64_from_all_jsons_visible,
            checkable=True,
            checked=False,
        )

        showParallelLineDist = action(
            self.tr("平行直線との距離調整を表示"),
            self._toggle_parallel_line_dist_visible,
            checkable=True,
            checked=False,
        )

        showTextBounding = action(
            self.tr("文字外接調整を表示"),
            self._toggle_text_bounding_visible,
            checkable=True,
            checked=False,
        )

        showStoreImageData = action(
            self.tr("JSONに画像データ保存を表示"),
            self._toggle_store_image_data_visible,
            checkable=True,
            checked=False,
        )

        showLineFit = action(
            self.tr("直線へのフィットを表示"),
            self._toggle_line_fit_visible,
            checkable=True,
            checked=False,
        )

        showDarkPixelMagnet = action(
            self.tr("カーソル配下色制御を表示"),
            self._toggle_dark_pixel_magnet_visible,
            checkable=True,
            checked=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="question.svg",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)

        # Line opacity widget
        lineOpacity = QtWidgets.QWidgetAction(self)
        lineOpacityBoxLayout = QtWidgets.QVBoxLayout()
        lineOpacityLabel = QtWidgets.QLabel(self.tr("線の\n透明度"))
        lineOpacityLabel.setAlignment(Qt.AlignCenter)
        lineOpacityBoxLayout.addWidget(lineOpacityLabel)
        lineOpacityBoxLayout.addWidget(self.lineOpacityWidget)
        lineOpacity.setDefaultWidget(QtWidgets.QWidget())
        lineOpacity.defaultWidget().setLayout(lineOpacityBoxLayout)
        self._lineOpacityAction = lineOpacity

        # Point opacity widget
        pointOpacity = QtWidgets.QWidgetAction(self)
        pointOpacityBoxLayout = QtWidgets.QVBoxLayout()
        pointOpacityLabel = QtWidgets.QLabel(self.tr("頂点の\n透明度"))
        pointOpacityLabel.setAlignment(Qt.AlignCenter)
        pointOpacityBoxLayout.addWidget(pointOpacityLabel)
        pointOpacityBoxLayout.addWidget(self.pointOpacityWidget)
        pointOpacity.setDefaultWidget(QtWidgets.QWidget())
        pointOpacity.defaultWidget().setLayout(pointOpacityBoxLayout)
        self._pointOpacityAction = pointOpacity

        # Fill opacity widget
        fillOpacity = QtWidgets.QWidgetAction(self)
        fillOpacityBoxLayout = QtWidgets.QVBoxLayout()
        fillOpacityLabel = QtWidgets.QLabel(self.tr("塗りの\n透明度"))
        fillOpacityLabel.setAlignment(Qt.AlignCenter)
        fillOpacityBoxLayout.addWidget(fillOpacityLabel)
        fillOpacityBoxLayout.addWidget(self.fillOpacityWidget)
        fillOpacity.setDefaultWidget(QtWidgets.QWidget())
        fillOpacity.defaultWidget().setLayout(fillOpacityBoxLayout)
        self._fillOpacityAction = fillOpacity

        # Line width widget
        lineWidth = QtWidgets.QWidgetAction(self)
        lineWidthBoxLayout = QtWidgets.QVBoxLayout()
        lineWidthLabel = QtWidgets.QLabel(self.tr("線の太さ"))
        lineWidthLabel.setAlignment(Qt.AlignCenter)
        lineWidthBoxLayout.addWidget(lineWidthLabel)
        lineWidthBoxLayout.addWidget(self.lineWidthWidget)
        lineWidth.setDefaultWidget(QtWidgets.QWidget())
        lineWidth.defaultWidget().setLayout(lineWidthBoxLayout)
        self._lineWidthAction = lineWidth

        # Point object size widget (point shape_type only)
        pointObjectSize = QtWidgets.QWidgetAction(self)
        pointObjectSizeBoxLayout = QtWidgets.QVBoxLayout()
        pointObjectSizeLabel = QtWidgets.QLabel(self.tr("点の大きさ"))
        pointObjectSizeLabel.setAlignment(Qt.AlignCenter)
        pointObjectSizeBoxLayout.addWidget(pointObjectSizeLabel)
        pointObjectSizeBoxLayout.addWidget(self.pointObjectSizeWidget)
        pointObjectSize.setDefaultWidget(QtWidgets.QWidget())
        pointObjectSize.defaultWidget().setLayout(pointObjectSizeBoxLayout)
        self._pointObjectSizeAction = pointObjectSize

        # Vertex size widget (polygon/rectangle vertices)
        vertexSize = QtWidgets.QWidgetAction(self)
        vertexSizeBoxLayout = QtWidgets.QVBoxLayout()
        vertexSizeLabel = QtWidgets.QLabel(self.tr("頂点の大きさ"))
        vertexSizeLabel.setAlignment(Qt.AlignCenter)
        vertexSizeBoxLayout.addWidget(vertexSizeLabel)
        vertexSizeBoxLayout.addWidget(self.vertexSizeWidget)
        vertexSize.setDefaultWidget(QtWidgets.QWidget())
        vertexSize.defaultWidget().setLayout(vertexSizeBoxLayout)
        self._vertexSizeAction = vertexSize

        # Save image data (base64) in JSON checkbox widget
        storeImageData = QtWidgets.QWidgetAction(self)
        storeImageDataBoxLayout = QtWidgets.QVBoxLayout()
        storeImageDataLabel = QtWidgets.QLabel(self.tr("JSONに\n画像データ保存"))
        storeImageDataLabel.setAlignment(Qt.AlignCenter)
        storeImageDataBoxLayout.addWidget(storeImageDataLabel)
        storeImageDataBoxLayout.addWidget(
            self.storeImageDataCheckbox, alignment=Qt.AlignCenter
        )
        storeImageData.setDefaultWidget(QtWidgets.QWidget())
        storeImageData.defaultWidget().setLayout(storeImageDataBoxLayout)
        storeImageData.setVisible(False)
        self._storeImageDataAction = storeImageData

        # Custom cursor checkbox widget
        customCursor = QtWidgets.QWidgetAction(self)
        customCursorBoxLayout = QtWidgets.QVBoxLayout()
        customCursorLabel = QtWidgets.QLabel(self.tr("カスタム\nカーソル"))
        customCursorLabel.setAlignment(Qt.AlignCenter)
        customCursorBoxLayout.addWidget(customCursorLabel)
        customCursorBoxLayout.addWidget(
            self.customCursorCheckbox, alignment=Qt.AlignCenter
        )
        customCursor.setDefaultWidget(QtWidgets.QWidget())
        customCursor.defaultWidget().setLayout(customCursorBoxLayout)

        # Right-click edit mode checkbox widget
        rightClickEdit = QtWidgets.QWidgetAction(self)
        rightClickEditBoxLayout = QtWidgets.QVBoxLayout()
        rightClickEditLabel = QtWidgets.QLabel(self.tr("右クリックで\n編集に切替"))
        rightClickEditLabel.setAlignment(Qt.AlignCenter)
        rightClickEditBoxLayout.addWidget(rightClickEditLabel)
        rightClickEditBoxLayout.addWidget(
            self.rightClickEditCheckbox, alignment=Qt.AlignCenter
        )
        rightClickEdit.setDefaultWidget(QtWidgets.QWidget())
        rightClickEdit.defaultWidget().setLayout(rightClickEditBoxLayout)

        # Skip delete confirmation checkbox widget
        skipDeleteConfirm = QtWidgets.QWidgetAction(self)
        skipDeleteConfirmBoxLayout = QtWidgets.QVBoxLayout()
        skipDeleteConfirmLabel = QtWidgets.QLabel(self.tr("ポリゴン削除\n確認不要"))
        skipDeleteConfirmLabel.setAlignment(Qt.AlignCenter)
        skipDeleteConfirmBoxLayout.addWidget(skipDeleteConfirmLabel)
        skipDeleteConfirmBoxLayout.addWidget(
            self.skipDeleteConfirmCheckbox, alignment=Qt.AlignCenter
        )
        skipDeleteConfirm.setDefaultWidget(QtWidgets.QWidget())
        skipDeleteConfirm.defaultWidget().setLayout(skipDeleteConfirmBoxLayout)

        # Skip save name confirmation checkbox widget
        skipSaveNameConfirm = QtWidgets.QWidgetAction(self)
        skipSaveNameConfirmBoxLayout = QtWidgets.QVBoxLayout()
        skipSaveNameConfirmLabel = QtWidgets.QLabel(self.tr("保存名称\n確認不要"))
        skipSaveNameConfirmLabel.setAlignment(Qt.AlignCenter)
        skipSaveNameConfirmBoxLayout.addWidget(skipSaveNameConfirmLabel)
        skipSaveNameConfirmBoxLayout.addWidget(
            self.skipSaveNameConfirmCheckbox, alignment=Qt.AlignCenter
        )
        skipSaveNameConfirm.setDefaultWidget(QtWidgets.QWidget())
        skipSaveNameConfirm.defaultWidget().setLayout(skipSaveNameConfirmBoxLayout)

        # Parallel line distance adjustment checkbox widget
        parallelLineDist = QtWidgets.QWidgetAction(self)
        parallelLineDistBoxLayout = QtWidgets.QVBoxLayout()
        parallelLineDistLabel = QtWidgets.QLabel(self.tr("平行直線との\n距離調整"))
        parallelLineDistLabel.setAlignment(Qt.AlignCenter)
        parallelLineDistBoxLayout.addWidget(parallelLineDistLabel)
        parallelLineDistBoxLayout.addWidget(
            self.parallelLineDistCheckbox, alignment=Qt.AlignCenter
        )
        parallelLineDist.setDefaultWidget(QtWidgets.QWidget())
        parallelLineDist.defaultWidget().setLayout(parallelLineDistBoxLayout)
        parallelLineDist.setVisible(False)
        self._parallelLineDistAction = parallelLineDist

        textBounding = QtWidgets.QWidgetAction(self)
        textBoundingBoxLayout = QtWidgets.QVBoxLayout()
        textBoundingLabel = QtWidgets.QLabel(self.tr("文字外接\n調整"))
        textBoundingLabel.setAlignment(Qt.AlignCenter)
        textBoundingBoxLayout.addWidget(textBoundingLabel)
        textBoundingBoxLayout.addWidget(
            self.textBoundingCheckbox, alignment=Qt.AlignCenter
        )
        textBounding.setDefaultWidget(QtWidgets.QWidget())
        textBounding.defaultWidget().setLayout(textBoundingBoxLayout)
        textBounding.setVisible(False)
        self._textBoundingAction = textBounding

        lineFit = QtWidgets.QWidgetAction(self)
        lineFitBoxLayout = QtWidgets.QVBoxLayout()
        lineFitLabel = QtWidgets.QLabel(self.tr("直線との\nフィット"))
        lineFitLabel.setAlignment(Qt.AlignCenter)
        lineFitBoxLayout.addWidget(lineFitLabel)
        lineFitBoxLayout.addWidget(
            self.lineFitCheckbox, alignment=Qt.AlignCenter
        )
        lineFit.setDefaultWidget(QtWidgets.QWidget())
        lineFit.defaultWidget().setLayout(lineFitBoxLayout)
        lineFit.setVisible(False)
        self._lineFitAction = lineFit

        darkPixelMagnet = QtWidgets.QWidgetAction(self)
        darkPixelMagnetBoxLayout = QtWidgets.QVBoxLayout()
        darkPixelMagnetLabel = QtWidgets.QLabel(self.tr("カーソル\n配下色制御"))
        darkPixelMagnetLabel.setAlignment(Qt.AlignCenter)
        darkPixelMagnetBoxLayout.addWidget(darkPixelMagnetLabel)
        darkPixelMagnetBoxLayout.addWidget(
            self.darkPixelMagnetCheckbox, alignment=Qt.AlignCenter
        )
        darkPixelMagnet.setDefaultWidget(QtWidgets.QWidget())
        darkPixelMagnet.defaultWidget().setLayout(darkPixelMagnetBoxLayout)
        darkPixelMagnet.setVisible(False)
        self._darkPixelMagnetAction = darkPixelMagnet

        autoFit = QtWidgets.QWidgetAction(self)
        autoFitBoxLayout = QtWidgets.QVBoxLayout()
        autoFitLabel = QtWidgets.QLabel(self.tr("矩形移動時\n自動フィット"))
        autoFitLabel.setAlignment(Qt.AlignCenter)
        autoFitBoxLayout.addWidget(autoFitLabel)
        autoFitBoxLayout.addWidget(
            self.autoFitCheckbox, alignment=Qt.AlignCenter
        )
        autoFit.setDefaultWidget(QtWidgets.QWidget())
        autoFit.defaultWidget().setLayout(autoFitBoxLayout)
        autoFit.setVisible(False)
        self._autoFitAction = autoFit

        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            lambda _: self._add_zoom(increment=1.1),
            shortcuts["zoom_in"],
            icon="magnifying-glass-minus.svg",
            tip=self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            lambda _: self._add_zoom(increment=0.9),
            shortcuts["zoom_out"],
            icon="magnifying-glass-plus.svg",
            tip=self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            self._set_zoom_to_original,
            shortcuts["zoom_to_original"],
            icon="image-square.svg",
            tip=self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            icon="frame-corners.svg",
            tip=self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            icon="frame-arrows-horizontal.svg",
            tip=self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "brightness-contrast.svg",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        self._zoom_mode = _ZoomMode.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            _ZoomMode.FIT_WINDOW: self.scaleFitWindow,
            _ZoomMode.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            _ZoomMode.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            icon="note-pencil.svg",
            tip=self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        changeSame = action(
            self.tr("Change to Same Label"),
            self._change_to_same_label,
            None,
            icon="tag.svg",
            tip=self.tr("Change label to the last used label"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            icon="paint-bucket.svg",
            tip=self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = types.SimpleNamespace(
            about=action(
                text=f"&About {__appname__}",
                slot=functools.partial(
                    QMessageBox.about,
                    self,
                    f"About {__appname__}",
                    f"""
<h3>{__appname__}</h3>
<p>Image Polygonal Annotation with Python</p>
<p>Version: {__version__}</p>
<p>Author: Kentaro Wada</p>
<p>
    <a href="https://labelme.io">Homepage</a> |
    <a href="https://labelme.io/docs">Documentation</a> |
    <a href="https://labelme.io/docs/troubleshoot">Troubleshooting</a>
</p>
<p>
    <a href="https://github.com/wkentaro/labelme">GitHub</a> |
    <a href="https://x.com/labelmeai">Twitter/X</a>
</p>
""",
                ),
            ),
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            exportFileList=exportFileList,
            progressStats=progressStats,
            toggle_keep_prev_brightness_contrast=action(
                text=self.tr("Keep Previous Brightness/Contrast"),
                slot=lambda: self._config.__setitem__(
                    "keep_prev_brightness_contrast",
                    not self._config["keep_prev_brightness_contrast"],
                ),
                checkable=True,
                checked=self._config["keep_prev_brightness_contrast"],
            ),
            delete=delete,
            edit=edit,
            changeSame=changeSame,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createMagicWandMode=createMagicWandMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            redo=redo,
            createPolygon3Mode=createPolygon3Mode,
            showCreatePolygon3=showCreatePolygon3,
            showCreateCircle=showCreateCircle,
            showCreateLine=showCreateLine,
            showCreateLineStrip=showCreateLineStrip,
            showCreateAiPolygon=showCreateAiPolygon,
            showCreateAiMask=showCreateAiMask,
            showCreateMagicWand=showCreateMagicWand,
            showRedo=showRedo,
            showStoreImageData=showStoreImageData,
            showParallelLineDist=showParallelLineDist,
            showTextBounding=showTextBounding,
            showLineFit=showLineFit,
            showDarkPixelMagnet=showDarkPixelMagnet,
            showAutoFit=showAutoFit,
            showLineOpacity=showLineOpacity,
            showPointOpacity=showPointOpacity,
            showFillOpacity=showFillOpacity,
            showLineWidth=showLineWidth,
            showPointObjectSize=showPointObjectSize,
            showVertexSize=showVertexSize,
            rectToTriangle=rectToTriangle,
            showRectToTriangle=showRectToTriangle,
            showDeleteAllShapes=showDeleteAllShapes,
            showRemoveHighIouShapes=showRemoveHighIouShapes,
            showCopyFromSpecifiedJson=showCopyFromSpecifiedJson,
            showCopyFromPrevJson=showCopyFromPrevJson,
            showAddBase64ToAllJsons=showAddBase64ToAllJsons,
            showRemoveBase64FromAllJsons=showRemoveBase64FromAllJsons,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            deleteAllShapes=deleteAllShapes,
            removeHighIouShapes=removeHighIouShapes,
            copyFromSpecifiedJson=copyFromSpecifiedJson,
            copyFromPrevJson=copyFromPrevJson,
            addBase64ToAllJsons=addBase64ToAllJsons,
            removeBase64FromAllJsons=removeBase64FromAllJsons,
        )
        self.on_shapes_present_actions = (saveAs, hideAll, showAll, toggleAll)

        self.draw_actions: list[tuple[str, QtWidgets.QAction]] = [
            ("polygon", createMode),
            ("polygon3", createPolygon3Mode),
            ("rectangle", createRectangleMode),
            ("point", createPointMode),
            ("circle", createCircleMode),
            ("line", createLineMode),
            ("linestrip", createLineStripMode),
            ("ai_polygon", createAiPolygonMode),
            ("ai_mask", createAiMaskMode),
        ]
        # Magic wand uses its own createMode, managed separately
        self._magic_wand_action = createMagicWandMode

        # Group zoom controls into a list for easier toggling.
        self.zoom_actions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.on_load_active_actions = (
            close,
            createMode,
            createPolygon3Mode,
            createRectangleMode,
            createCircleMode,
            createLineMode,
            createPointMode,
            createMagicWandMode,
            createLineStripMode,
            createAiPolygonMode,
            createAiMaskMode,
            brightnessContrast,
            deleteAllShapes,
            removeHighIouShapes,
            copyFromSpecifiedJson,
            copyFromPrevJson,
            addBase64ToAllJsons,
            removeBase64FromAllJsons,
        )
        # menu shown at right click
        self.context_menu_actions = (
            changeSame,
            edit,
            duplicate,
            delete,
            undo,
            redo,
            removePoint,
        )
        # XXX: need to add some actions here to activate the shortcut
        self.edit_menu_actions = (
            edit,
            copy,
            paste,
            None,
            undoLastPoint,
            None,
            removePoint,
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = types.SimpleNamespace(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                close,
                deleteFile,
                None,
                open_config,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help, self.actions.about))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
                self.actions.toggle_keep_prev_brightness_contrast,
                None,
                showCreateMagicWand,
                showCreatePolygon3,
                showCreateCircle,
                showCreateLine,
                showCreateLineStrip,
                showCreateAiPolygon,
                showCreateAiMask,
                None,
                showRedo,
                showRectToTriangle,
                showDeleteAllShapes,
                showRemoveHighIouShapes,
                showCopyFromSpecifiedJson,
                showCopyFromPrevJson,
                showAddBase64ToAllJsons,
                showRemoveBase64FromAllJsons,
                None,
                showLineOpacity,
                showPointOpacity,
                showFillOpacity,
                showLineWidth,
                showPointObjectSize,
                showVertexSize,
                None,
                showStoreImageData,
                showLineFit,
                showParallelLineDist,
                showTextBounding,
                showDarkPixelMagnet,
                showAutoFit,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        self.canvas._undo_action = self.actions.undo
        self.canvas._redo_action = self.actions.redo
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        self._ai_assisted_annotation_widget: AiAssistedAnnotationWidget = (
            AiAssistedAnnotationWidget(
                default_model=self._config["ai"]["default"],
                on_model_changed=self.canvas.set_ai_model_name,
                parent=self,
            )
        )
        self._ai_assisted_annotation_widget.setEnabled(False)
        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(self._ai_assisted_annotation_widget)

        self._ai_text_to_annotation_widget: AiTextToAnnotationWidget = (
            AiTextToAnnotationWidget(on_submit=self._submit_ai_prompt, parent=self)
        )
        self._ai_text_to_annotation_widget.setEnabled(False)
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_text_to_annotation_widget)

        tools_toolbar = ToolBar(
                title="Tools",
                actions=[
                    open_,
                    opendir,
                    openPrevImg,
                    openNextImg,
                    save,
                    deleteFile,
                    exportFileList,
                    progressStats,
                    None,
                    editMode,
                    duplicate,
                    delete,
                    undo,
                    redo,
                    rectToTriangle,
                    deleteAllShapes,
                    removeHighIouShapes,
                    copyFromSpecifiedJson,
                    copyFromPrevJson,
                    addBase64ToAllJsons,
                    removeBase64FromAllJsons,
                    None,
                    fitWindow,
                    zoom,
                    lineOpacity,
                    pointOpacity,
                    fillOpacity,
                    lineWidth,
                    pointObjectSize,
                    vertexSize,
                    None,
                    customCursor,
                    rightClickEdit,
                    skipDeleteConfirm,
                    skipSaveNameConfirm,
                    storeImageData,
                    lineFit,
                    parallelLineDist,
                    textBounding,
                    darkPixelMagnet,
                    autoFit,
                    None,
                    selectAiModel,
                    None,
                    ai_prompt_action,
                ],
                font_base=self.font(),
        )
        self.addToolBar(Qt.TopToolBarArea, tools_toolbar)
        self._tools_toolbar = tools_toolbar
        # Build left toolbar: insert magic wand after point
        left_actions = []
        for mode, act in self.draw_actions:
            left_actions.append(act)
            if mode == "point":
                left_actions.append(self._magic_wand_action)
        create_toolbar = ToolBar(
            title="CreateShapeTools",
            actions=left_actions,
            orientation=Qt.Vertical,
            button_style=Qt.ToolButtonTextUnderIcon,
            font_base=self.font(),
        )
        self.addToolBar(Qt.LeftToolBarArea, create_toolbar)
        self._create_toolbar = create_toolbar

        self.status_left = QtWidgets.QLabel(self.tr("%s started.") % __appname__)
        self.status_right = StatusStats()
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addWidget(self.status_right, 0)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.labelFile: LabelFile | None = None
        self.imagePath: str | None = None
        self.recentFiles: list[str] = []
        self.maxRecent = 7
        self._other_data = None
        self.zoom_level = 100
        self.fit_window = False
        self._zoom_values = {}
        self._brightness_contrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if self._config["file_search"]:
            self.fileSearch.setText(self._config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self._prev_opened_dir = self.settings.value("lastOpenedDir", None) or None
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(900, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        layout_version = self.settings.value("window/state_version", 0, type=int)
        state = self.settings.value("window/state", QtCore.QByteArray())
        self._force_right_dock_layout = False
        self._skip_restore_state = False
        if layout_version < 3:
            # Reset stale dock layout once to ensure splitters exist (notably on macOS).
            state = QtCore.QByteArray()
            self.settings.setValue("window/state_version", 3)
            self.settings.setValue("window/state", QtCore.QByteArray())
            self._force_right_dock_layout = True
            self._skip_restore_state = True
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry'])
        if not self._skip_restore_state:
            self.restoreState(state)
        # Ensure dock splitters exist even if a previous saved state overwrote them.
        # On macOS, apply after the window is shown to avoid layout overrides.
        self._did_force_right_dock_splits = False
        QtCore.QTimer.singleShot(0, self._ensure_right_dock_splits)

        # Restore opacity and line width settings
        lineOpacity = self.settings.value("canvas/lineOpacity", 70, type=int)
        pointOpacity = self.settings.value("canvas/pointOpacity", 50, type=int)
        fillOpacity = self.settings.value("canvas/fillOpacity", 80, type=int)
        lineWidth = self.settings.value("canvas/lineWidth", 6, type=int)
        pointObjectSize = self.settings.value(
            "canvas/pointObjectSize", 24, type=int
        )
        vertexSize = self.settings.value("canvas/vertexSize", 8, type=int)
        self.lineOpacityWidget.setValue(lineOpacity)
        self.pointOpacityWidget.setValue(pointOpacity)
        self.fillOpacityWidget.setValue(fillOpacity)
        self.lineWidthWidget.setValue(lineWidth)
        self.pointObjectSizeWidget.setValue(pointObjectSize)
        self.vertexSizeWidget.setValue(vertexSize)
        # Visibility of adjustment widgets (default visible).
        for key, action_, toggle in (
            ("view/showLineOpacity", "showLineOpacity",
             self._toggle_line_opacity_visible),
            ("view/showPointOpacity", "showPointOpacity",
             self._toggle_point_opacity_visible),
            ("view/showFillOpacity", "showFillOpacity",
             self._toggle_fill_opacity_visible),
            ("view/showLineWidth", "showLineWidth",
             self._toggle_line_width_visible),
            ("view/showPointObjectSize", "showPointObjectSize",
             self._toggle_point_object_size_visible),
            ("view/showVertexSize", "showVertexSize",
             self._toggle_vertex_size_visible),
        ):
            visible = self.settings.value(key, True, type=bool)
            getattr(self.actions, action_).setChecked(visible)
            toggle(visible)
        storeImageDataEnabled = self.settings.value(
            "canvas/storeImageData", self._config["store_data"], type=bool
        )
        self.storeImageDataCheckbox.blockSignals(True)
        self.storeImageDataCheckbox.setChecked(storeImageDataEnabled)
        self.storeImageDataCheckbox.blockSignals(False)
        self._config["store_data"] = bool(storeImageDataEnabled)
        self.actions.saveWithImageData.setChecked(bool(storeImageDataEnabled))
        showStoreImageDataEnabled = self.settings.value(
            "view/showStoreImageData", False, type=bool
        )
        self.actions.showStoreImageData.setChecked(showStoreImageDataEnabled)
        self._toggle_store_image_data_visible(showStoreImageDataEnabled)

        customCursorEnabled = self.settings.value(
            "canvas/customCursor", False, type=bool
        )
        self.customCursorCheckbox.setChecked(customCursorEnabled)
        rightClickEditEnabled = self.settings.value(
            "canvas/rightClickEdit", False, type=bool
        )
        self.rightClickEditCheckbox.setChecked(rightClickEditEnabled)
        skipDeleteConfirmEnabled = self.settings.value(
            "canvas/skipDeleteConfirm", False, type=bool
        )
        self.skipDeleteConfirmCheckbox.setChecked(skipDeleteConfirmEnabled)
        skipSaveNameConfirmEnabled = self.settings.value(
            "canvas/skipSaveNameConfirm", False, type=bool
        )
        self.skipSaveNameConfirmCheckbox.setChecked(skipSaveNameConfirmEnabled)
        parallelLineDistEnabled = self.settings.value(
            "canvas/parallelLineDist", False, type=bool
        )
        self.parallelLineDistCheckbox.setChecked(parallelLineDistEnabled)
        for key, toggle_fn in (
            ("showCreatePolygon3", self._toggle_create_polygon3_visible),
            ("showCreateCircle", self._toggle_create_circle_visible),
            ("showCreateLine", self._toggle_create_line_visible),
            ("showCreateLineStrip", self._toggle_create_linestrip_visible),
            ("showCreateAiPolygon", self._toggle_create_ai_polygon_visible),
            ("showCreateAiMask", self._toggle_create_ai_mask_visible),
            ("showCreateMagicWand", self._toggle_create_magic_wand_visible),
        ):
            enabled = self.settings.value(f"view/{key}", False, type=bool)
            getattr(self.actions, key).setChecked(enabled)
            toggle_fn(enabled)
        showRedoEnabled = self.settings.value("view/showRedo", False, type=bool)
        self.actions.showRedo.setChecked(showRedoEnabled)
        self._toggle_redo_visible(showRedoEnabled)
        for key, action_, toggle, default_visible in (
            ("view/showRectToTriangle", "showRectToTriangle",
             self._toggle_rect_to_triangle_visible, True),
            ("view/showDeleteAllShapes", "showDeleteAllShapes",
             self._toggle_delete_all_shapes_visible, True),
            ("view/showRemoveHighIouShapes", "showRemoveHighIouShapes",
             self._toggle_remove_high_iou_visible, True),
            ("view/showCopyFromSpecifiedJson", "showCopyFromSpecifiedJson",
             self._toggle_copy_from_specified_json_visible, True),
            ("view/showCopyFromPrevJson", "showCopyFromPrevJson",
             self._toggle_copy_from_prev_json_visible, True),
            ("view/showAddBase64ToAllJsons", "showAddBase64ToAllJsons",
             self._toggle_add_base64_to_all_jsons_visible, False),
            ("view/showRemoveBase64FromAllJsons", "showRemoveBase64FromAllJsons",
             self._toggle_remove_base64_from_all_jsons_visible, False),
        ):
            visible = self.settings.value(key, default_visible, type=bool)
            getattr(self.actions, action_).setChecked(visible)
            toggle(visible)
        showParallelLineDistEnabled = self.settings.value(
            "view/showParallelLineDist", False, type=bool
        )
        self.actions.showParallelLineDist.setChecked(showParallelLineDistEnabled)
        self._toggle_parallel_line_dist_visible(showParallelLineDistEnabled)
        textBoundingEnabled = self.settings.value(
            "canvas/textBounding", False, type=bool
        )
        self.textBoundingCheckbox.setChecked(textBoundingEnabled)
        showTextBoundingEnabled = self.settings.value(
            "view/showTextBounding", False, type=bool
        )
        self.actions.showTextBounding.setChecked(showTextBoundingEnabled)
        self._toggle_text_bounding_visible(showTextBoundingEnabled)
        lineFitEnabled = self.settings.value(
            "canvas/lineFit", False, type=bool
        )
        self.lineFitCheckbox.blockSignals(True)
        self.lineFitCheckbox.setChecked(lineFitEnabled)
        self.lineFitCheckbox.blockSignals(False)
        # Mutual exclusion: if both line fit and parallel line are on,
        # keep line fit and turn off parallel line.
        if lineFitEnabled and self.parallelLineDistCheckbox.isChecked():
            self.parallelLineDistCheckbox.setChecked(False)
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setLineFitEnabled(lineFitEnabled)
        showLineFitEnabled = self.settings.value(
            "view/showLineFit", False, type=bool
        )
        self.actions.showLineFit.setChecked(showLineFitEnabled)
        self._toggle_line_fit_visible(showLineFitEnabled)
        darkPixelMagnetEnabled = self.settings.value(
            "canvas/darkPixelMagnet", False, type=bool
        )
        self.darkPixelMagnetCheckbox.setChecked(darkPixelMagnetEnabled)
        showDarkPixelMagnetEnabled = self.settings.value(
            "view/showDarkPixelMagnet", False, type=bool
        )
        self.actions.showDarkPixelMagnet.setChecked(showDarkPixelMagnetEnabled)
        self._toggle_dark_pixel_magnet_visible(showDarkPixelMagnetEnabled)
        autoFitEnabled = self.settings.value(
            "canvas/autoFit", False, type=bool
        )
        self.autoFitCheckbox.setChecked(autoFitEnabled)
        showAutoFitEnabled = self.settings.value(
            "view/showAutoFit", False, type=bool
        )
        self.actions.showAutoFit.setChecked(showAutoFitEnabled)
        self._toggle_auto_fit_visible(showAutoFitEnabled)

        if filename:
            if osp.isdir(filename):
                self._import_images_from_dir(root_dir=filename)
                self._open_next_image()
            else:
                self._load_file(filename=filename)
        else:
            self.filename = None
            # First launch (or stale prev dir): prompt the user to pick a
            # folder once the main window has been shown. Only when the app
            # asked for it (__main__), never for an embedded/test window —
            # a modal dialog nobody can close would hang the caller.
            if prompt_for_dir and not (
                self._prev_opened_dir and osp.exists(self._prev_opened_dir)
            ):
                QtCore.QTimer.singleShot(0, self._open_dir_with_dialog)

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self._paint_canvas)

        self.populateModeActions()

    def _load_config(
        self, config_file: Path | None, config_overrides: dict | None
    ) -> tuple[Path | None, dict]:
        try:
            config = load_config(
                config_file=config_file, config_overrides=config_overrides or {}
            )
        except ValueError as e:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle(self.tr("Configuration Errors"))
            msg_box.setText(
                self.tr(
                    "Errors were found while loading the configuration. "
                    "Please review the errors below and reload your configuration or "
                    "ignore the erroneous lines."
                )
            )
            msg_box.setInformativeText(str(e))
            msg_box.setStandardButtons(QMessageBox.Ignore)
            msg_box.setModal(False)
            msg_box.show()

            config_file = None
            config_overrides = {}
            config = load_config(
                config_file=config_file, config_overrides=config_overrides
            )
        return config_file, config

    def _ensure_right_dock_splits(self) -> None:
        dock_order = [
            self.navigator_dock,
            self.update_distribution_dock,
            self.flag_dock,
            self.label_dock,
            self.shape_dock,
            self.file_dock,
        ]
        if self._force_right_dock_layout:
            for dock in dock_order:
                self.removeDockWidget(dock)
            if self.navigator_dock is not None:
                self.navigator_dock.deleteLater()
            if self.navigator is not None:
                self.navigator.deleteLater()
            self.navigator, self.navigator_dock = self._create_navigator_dock()
            dock_order[0] = self.navigator_dock
            if self.update_distribution_dock is not None:
                self.update_distribution_dock.deleteLater()
            if self.update_distribution is not None:
                self.update_distribution.deleteLater()
            self.update_distribution, self.update_distribution_dock = (
                self._create_update_distribution_dock()
            )
            dock_order[1] = self.update_distribution_dock
        visible_docks = [dock for dock in dock_order if dock.isVisible()]
        if len(visible_docks) < 2:
            visible_docks = dock_order
        self.addDockWidget(Qt.RightDockWidgetArea, visible_docks[0])
        for dock in visible_docks[1:]:
            self.addDockWidget(Qt.RightDockWidgetArea, dock, Qt.Vertical)
        self.resizeDocks(
            [
                self.navigator_dock,
                self.update_distribution_dock,
                self.flag_dock,
                self.label_dock,
                self.shape_dock,
                self.file_dock,
            ],
            [180, 180, 140, 140, 180, 180],
            Qt.Vertical,
        )

    def showEvent(self, event):
        super().showEvent(event)
        if not self._did_force_right_dock_splits:
            self._ensure_right_dock_splits()
            self._did_force_right_dock_splits = True

    def _create_navigator_dock(
        self,
    ) -> tuple[NavigatorWidget, QtWidgets.QDockWidget]:
        navigator = NavigatorWidget()
        navigator_dock = QtWidgets.QDockWidget(self.tr("Navigator"), self)
        navigator_dock.setObjectName("Navigator")
        navigator_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        # Wrap in QScrollArea to allow dock resizing on macOS
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(navigator)
        scroll_area.setWidgetResizable(True)  # Navigator stretches with dock
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        navigator_dock.setWidget(scroll_area)
        navigator.viewportChangeRequested.connect(self._onNavigatorViewportChange)
        return navigator, navigator_dock

    def _create_update_distribution_dock(
        self,
    ) -> tuple[UpdateDistributionWidget, QtWidgets.QDockWidget]:
        update_dist = UpdateDistributionWidget()
        update_dist_dock = QtWidgets.QDockWidget(self.tr("Update Distribution"), self)
        update_dist_dock.setObjectName("UpdateDistribution")
        update_dist_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        # Wrap in QScrollArea to allow dock resizing on macOS
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(update_dist)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        update_dist_dock.setWidget(scroll_area)
        update_dist.viewportChangeRequested.connect(self._onNavigatorViewportChange)
        return update_dist, update_dist_dock

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        self.menus.edit.clear()
        actions = (
            *[draw_action for _, draw_action in self.draw_actions],
            self.actions.editMode,
            *self.edit_menu_actions,
        )
        utils.addActions(self.menus.edit, actions)

    def _get_window_title(self, dirty: bool) -> str:
        window_title: str = __appname__
        if self.imagePath:
            if self._prev_opened_dir:
                # Directory mode: {dir_name}/{filename}
                dir_name = osp.basename(self._prev_opened_dir)
                file_name = osp.basename(self.imagePath)
                display_path = f"{dir_name}/{file_name}"
            else:
                # File mode: {filename}
                display_path = osp.basename(self.imagePath)
            window_title = f"{window_title} - {display_path}"
            if not self.image.isNull():
                window_title = (
                    f"{window_title} "
                    f"(w{self.image.width()} x h{self.image.height()})"
                )
            if self.fileListWidget.count() and self.fileListWidget.currentItem():
                window_title = (
                    f"{window_title} "
                    f"[{self.fileListWidget.currentRow() + 1}"
                    f"/{self.fileListWidget.count()}]"
                )
            # Add JSON last modified time
            label_file = self._labelFileForImage(self.imagePath)
            if osp.exists(label_file):
                import datetime
                mtime = osp.getmtime(label_file)
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                window_title = f"{window_title} | {self.tr('Last saved')}: {mtime_str}"
        if dirty:
            window_title = f"{window_title} | {self.tr('Editing')}"
        return window_title

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.actions.redo.setEnabled(self.canvas.isShapeRedoable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            assert self.imagePath
            label_file = self._labelFileForImage(self.imagePath)
            if self.saveLabels(label_file):
                return
            # Saving failed (saveLabels already reported why): keep the
            # document dirty so the work is not silently lost.
            logger.warning("auto save failed: {!r}", label_file)
        self._is_changed = True
        self.actions.save.setEnabled(True)
        self.setWindowTitle(self._get_window_title(dirty=True))
        # The docks redraw every shape; with thousands of them that is the
        # pause felt when letting go of the mouse. Coalesce it to one update
        # once the interaction is over instead of one per edit.
        self._scheduleDockShapeRefresh()
        self._updateLabelListBackgrounds()

    def _scheduleDockShapeRefresh(self) -> None:
        if self._dock_refresh_pending:
            return
        self._dock_refresh_pending = True
        QtCore.QTimer.singleShot(0, self._refreshDockShapes)

    def _refreshDockShapes(self) -> None:
        self._dock_refresh_pending = False
        self.navigator.setShapes(self.canvas.shapes)
        self.update_distribution.setShapes(self.canvas.shapes)

    def _updateLabelListBackgrounds(self):
        """Update label list item backgrounds based on shape modification status."""
        cleared = QtGui.QBrush()
        for row in range(self.labelList._model.rowCount()):
            item = self.labelList._model.item(row)
            if item:
                shape = item.shape()
                if not shape:
                    continue
                brush = (
                    cleared
                    if getattr(shape, "modified_at", None)
                    else self.SHAPE_UNMODIFIED_COLOR
                )
                # setBackground always emits itemChanged, which reaches
                # labelItemChanged -> canvas.setShapeVisible -> a canvas
                # repaint. With thousands of labels that fires on every edit,
                # so only touch the rows that actually change.
                if item.background() != brush:
                    item.setBackground(brush)

    def setClean(self):
        self._is_changed = False
        self.actions.save.setEnabled(False)
        for _, action in self.draw_actions:
            action.setEnabled(True)
        self.setWindowTitle(self._get_window_title(dirty=False))

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.zoom_actions:
            z.setEnabled(value)
        for action in self.on_load_active_actions:
            action.setEnabled(value)
        if value:
            # Disable "1つ上のJSON" when on the first file
            if self.fileListWidget.currentRow() <= 0:
                self.actions.copyFromPrevJson.setEnabled(False)
            # Re-apply fit-button disabling (zoom_actions were bulk-enabled)
            self._sync_fit_actions()

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def show_status_message(self, message, delay=500):
        self.statusBar().showMessage(message, delay)

    def _submit_ai_prompt(self, _) -> None:
        if (
            self.canvas.createMode
            not in _AI_TEXT_TO_ANNOTATION_CREATE_MODE_TO_SHAPE_TYPE
        ):
            logger.warning("Unsupported createMode=%r", self.canvas.createMode)
            return
        shape_type: Literal["rectangle", "polygon", "mask"] = (
            _AI_TEXT_TO_ANNOTATION_CREATE_MODE_TO_SHAPE_TYPE[self.canvas.createMode]
        )

        texts = self._ai_text_to_annotation_widget.get_text_prompt().split(",")

        model_name: str = self._ai_text_to_annotation_widget.get_model_name()
        model_type = osam.apis.get_model_type_by_name(model_name)
        if not (_is_already_downloaded := model_type.get_size() is not None):
            if not download_ai_model(model_name=model_name, parent=self):
                return
        if (
            self._text_osam_session is None
            or self._text_osam_session.model_name != model_name
        ):
            self._text_osam_session = OsamSession(model_name=model_name)

        boxes, scores, labels, masks = bbox_from_text.get_bboxes_from_texts(
            session=self._text_osam_session,
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            image_id=str(hash(self.imagePath)),
            texts=texts,
        )

        SCORE_FOR_EXISTING_SHAPE: float = 1.01
        for shape in self.canvas.shapes:
            if shape.shape_type != shape_type or shape.label not in texts:
                continue
            points: NDArray[np.float64] = np.array(
                [[p.x(), p.y()] for p in shape.points]
            )
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [SCORE_FOR_EXISTING_SHAPE]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels, indices = bbox_from_text.nms_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_text_to_annotation_widget.get_iou_threshold(),
            score_threshold=self._ai_text_to_annotation_widget.get_score_threshold(),
            max_num_detections=100,
        )

        is_new = scores != SCORE_FOR_EXISTING_SHAPE
        boxes = boxes[is_new]
        scores = scores[is_new]
        labels = labels[is_new]
        indices = indices[is_new]

        if masks is not None:
            masks = masks[indices]
        del indices

        shapes: list[Shape] = bbox_from_text.get_shapes_from_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
            masks=masks,
            shape_type=shape_type,
        )

        self.canvas.storeShapes()
        self._load_shapes(shapes, replace=False)
        self.setDirty()

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self._other_data = None
        self.canvas.resetState()
        self.actions.redo.setEnabled(False)

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        if self.canvas.drawing() and self.canvas.current:
            self.canvas.undoLastPoint()
            self.actions.redo.setEnabled(len(self.canvas._undone_points) > 0)
            return
        self.canvas.restoreShape()
        redo_stack = list(self.canvas.shapesRedoStack)
        self.labelList.clear()
        self._load_shapes(self.canvas.shapes)
        self.canvas.shapesRedoStack = redo_stack
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.actions.redo.setEnabled(self.canvas.isShapeRedoable)

    def redoShapeEdit(self):
        if self.canvas.drawing() and self.canvas.current:
            self.canvas.redoLastPoint()
            self.actions.redo.setEnabled(len(self.canvas._undone_points) > 0)
            return
        self.canvas.redoShape()
        redo_stack = list(self.canvas.shapesRedoStack)
        self.labelList.clear()
        self._load_shapes(self.canvas.shapes)
        self.canvas.shapesRedoStack = redo_stack
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.actions.redo.setEnabled(self.canvas.isShapeRedoable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(True)
        if drawing:
            self.actions.redo.setEnabled(False)
        else:
            self.actions.redo.setEnabled(self.canvas.isShapeRedoable)
        # delete/duplicate/copy: only enable if not drawing AND shapes are selected
        n_selected = len(self.canvas.selectedShapes) if not drawing else 0
        self.actions.delete.setEnabled(n_selected > 0)
        self.actions.duplicate.setEnabled(n_selected > 0)
        self.actions.copy.setEnabled(n_selected > 0)

    def _onEditModeChanged(self, edit: bool) -> None:
        """Handle edit mode change from canvas (e.g., right-click to switch)."""
        if edit:
            for _, draw_action in self.draw_actions:
                draw_action.setEnabled(True)
            self._magic_wand_action.setEnabled(True)
        self.actions.editMode.setEnabled(not edit)

    def _switch_canvas_mode(
        self, edit: bool = True, createMode: str | None = None
    ) -> None:
        # Cancel any in-progress drawing first: an unfinished shape must not
        # leak into the next mode (leftover polygon preview after switching
        # to rectangle, stale `current` blocking point-creation clicks, etc.)
        self.canvas.cancelDrawing()
        self.canvas.setEditing(edit)
        if createMode is not None:
            self.canvas.createMode = createMode
        if edit:
            for _, draw_action in self.draw_actions:
                draw_action.setEnabled(True)
            self._magic_wand_action.setEnabled(True)
        else:
            for draw_mode, draw_action in self.draw_actions:
                draw_action.setEnabled(createMode != draw_mode)
            self._magic_wand_action.setEnabled(createMode != "magic_wand")
        self.actions.editMode.setEnabled(not edit)
        self._ai_text_to_annotation_widget.setEnabled(
            not edit and createMode in _AI_TEXT_TO_ANNOTATION_CREATE_MODE_TO_SHAPE_TYPE
        )
        self._ai_assisted_annotation_widget.setEnabled(
            not edit and createMode in ("ai_polygon", "ai_mask")
        )

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, f"&{i + 1} {QtCore.QFileInfo(f).fileName()}", self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)  # type: ignore[attr-defined,union-attr]
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def _change_to_same_label(self):
        """Change selected shapes' labels to the last used label."""
        if self._last_label is None:
            return

        items = self.labelList.selectedItems()
        if not items:
            return

        self.canvas.storeShapes()
        for item in items:
            shape = item.shape()
            old_label = shape.label
            shape.label = self._last_label
            # Update label list item
            item.setText(
                shape.label
                if shape.group_id is None
                else f"{shape.label} ({shape.group_id})"
            )
            self._update_shape_color(shape)
        self.setDirty()

    def _update_pending_draw_label(self):
        """Keep canvas pending draw label in sync for dark pixel magnet."""
        label = None
        items = self.uniqLabelList.selectedItems()
        if items:
            label = items[0].data(Qt.UserRole)
        elif self._last_label:
            label = self._last_label
        self.canvas.setPendingDrawLabel(label)

    def _update_change_same_action(self):
        """Update changeSame action text and enabled state."""
        if self._last_label:
            self.actions.changeSame.setText(
                self.tr("Change to '%s'") % self._last_label
            )
            # Enabled when there's a last label AND shapes are selected
            n_selected = len(self.canvas.selectedShapes)
            self.actions.changeSame.setEnabled(n_selected > 0)
        else:
            self.actions.changeSame.setText(
                self.tr("Change to '%s'") % "---"
            )
            self.actions.changeSame.setEnabled(False)

    def _edit_label(self, value=None):
        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
            flags_disabled=not edit_flags,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()  # type: ignore[no-redef]

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            if shape.group_id is None:
                r, g, b = shape.fill_color.getRgb()[:3]
                item.setText(
                    f"{html.escape(shape.label)} "
                    f'<font color="#{r:02x}{g:02x}{b:02x}">●</font>'
                )
            else:
                item.setText(f"{shape.label} ({shape.group_id})")
            self.setDirty()
            if self.uniqLabelList.find_label_item(shape.label) is None:
                self.uniqLabelList.add_label_item(
                    label=shape.label, color=self._get_rgb_by_label(label=shape.label)
                )
        # Update last used label for changeSame action
        if edit_text and text:
            self._last_label = text
            self._update_change_same_action()

    def fileSearchChanged(self):
        self._import_images_from_dir(
            root_dir=self._prev_opened_dir, pattern=self.fileSearch.text()
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self._can_continue():
            return

        # Use stored full path (UserRole) if available, else fall back to text
        file_path = item.data(Qt.UserRole)
        if not file_path:
            file_path = item.text()
        currIndex = self.imageList.index(file_path)
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self._load_file(filename)
        self._update_nav_button_state()

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        # A flag rather than disconnecting the signal: a nested call would
        # disconnect twice and raise, and the flag also survives re-entry
        # through any other emitter. Pushing our own selection into the label
        # list is what triggers re-entry, so the nested request is an echo of
        # the selection we are already applying: drop it. (Re-applying it
        # instead would let two handlers that each ask for the other's shape
        # spin this loop forever on the GUI thread.)
        if self._syncing_label_selection:
            logger.debug("ignoring re-entrant shape selection change")
            return
        self._syncing_label_selection = True
        try:
            self._apply_shape_selection(list(selected_shapes))
        finally:
            self._syncing_label_selection = False

    def _apply_shape_selection(self, selected_shapes):
        # Clear every previous flag and set every new one, exactly as before:
        # a shape whose flag drifted out of sync gets repaired.
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.canvas.selectedShapes = selected_shapes
        for shape in selected_shapes:
            shape.selected = True
        # One pass over the rows instead of a full scan per shape
        # (findItemByShape was O(rows) inside a loop over the selection).
        by_shape = self.labelList.itemsByShape()
        items = [
            by_shape[shape] for shape in selected_shapes if shape in by_shape
        ]
        self.labelList.selectOnlyItems(items)
        # Scroll to each item in order, as before, but repaint the list once
        # at the end instead of after every step. Restore whatever the update
        # state was, in case a caller had it disabled.
        updates_were_enabled = self.labelList.updatesEnabled()
        self.labelList.setUpdatesEnabled(False)
        try:
            for item in items:
                self.labelList.scrollToItem(item)
        finally:
            self.labelList.setUpdatesEnabled(updates_were_enabled)
        # Only the selection changed, so the hover order can be re-derived
        # from the cached area order instead of measuring every shape again.
        self.canvas.updateSelectionOrder()
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)
        self.actions.rectToTriangle.setEnabled(
            any(s.shape_type == "rectangle" for s in selected_shapes)
        )
        self._update_change_same_action()

    def addLabel(self, shape):
        label_list_item, x, y = self._createLabelItem(shape)

        # Insert in sorted order (key1: y, key2: x)
        insert_row = 0
        for row in range(self.labelList._model.rowCount()):
            item = self.labelList._model.item(row)
            if item:
                other_shape = item.shape()
                if other_shape and other_shape.points:
                    other_x = min(p.x() for p in other_shape.points)
                    other_y = min(p.y() for p in other_shape.points)
                    if (y, x) < (other_y, other_x):
                        break
            insert_row = row + 1
        self.labelList._model.insertRow(insert_row, label_list_item)
        self._applyLabelItemText(shape, label_list_item, x, y)

    def _addLabelsSorted(self, shapes) -> bool:
        """Add many labels at once, ordered like repeated sorted inserts.

        addLabel scans every existing row (recomputing its shape's min x/y)
        to find the insert position, which is O(n^2 * vertices) when a whole
        file is loaded or an undo reloads every shape. Sorting once and
        appending is equivalent *while the list starts empty*: the scan
        inserts after all rows with an equal key, i.e. exactly what a stable
        sort produces. Returns False when the caller must use addLabel
        (non-empty list, e.g. keep-previous shapes or a user-reordered list,
        and shapes without points, which the scan treats specially).
        """
        if self.labelList._model.rowCount() != 0:
            return False
        if any(not shape.points for shape in shapes):
            return False
        prepared = []
        for shape in shapes:
            item, x, y = self._createLabelItem(shape)
            prepared.append(((y, x), item, shape, x, y))
        prepared.sort(key=lambda entry: entry[0])  # stable: ties keep order
        model = self.labelList._model
        for _, item, shape, x, y in prepared:
            model.appendRow(item)
            self._applyLabelItemText(shape, item, x, y)
        return True

    def _applyLabelItemText(self, shape, label_list_item, x, y) -> None:
        if shape.group_id is None:
            text = shape.label
        else:
            text = f"{shape.label} ({shape.group_id})"
        self._update_shape_color(shape)
        r, g, b = shape.fill_color.getRgb()[:3]
        label_list_item.setText(
            f'{html.escape(text)} <font color="#{r:02x}{g:02x}{b:02x}">●</font>'
            f" ({int(x)},{int(y)})"
        )

    def _createLabelItem(self, shape):
        """Build the label-list row for a shape (everything addLabel does
        except choosing where to insert it). Returns (item, x, y)."""
        if shape.group_id is None:
            text = shape.label
        else:
            text = f"{shape.label} ({shape.group_id})"

        # Get shape position (bounding rect top-left)
        if shape.points:
            x = min(p.x() for p in shape.points)
            y = min(p.y() for p in shape.points)
        else:
            x, y = 0, 0

        label_list_item = LabelListWidgetItem(text, shape)

        # Highlight shapes without modification timestamp
        if not getattr(shape, 'modified_at', None):
            label_list_item.setBackground(self.SHAPE_UNMODIFIED_COLOR)

        if self.uniqLabelList.find_label_item(shape.label) is None:
            # Insert first with placeholder color, then update with correct
            # index-based color (sorted insert may shift the final index)
            self.uniqLabelList.add_label_item(
                label=shape.label, color=(0, 255, 0),
                sorted_insert=True,
            )
            # Now the label has its final index — update stored color
            item = self.uniqLabelList.find_label_item(shape.label)
            if item is not None:
                rgb = self._get_rgb_by_label(label=shape.label)
                item.setData(Qt.UserRole + 1, rgb)
                idx = self.uniqLabelList.get_label_index(shape.label)
                if idx is not None:
                    self.uniqLabelList._update_item_text(item, idx)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.on_shapes_present_actions:
            action.setEnabled(True)

        return label_list_item, x, y

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        # Apply current opacity settings
        lineOpacity = self.lineOpacityWidget.value()
        lineAlpha = int((100 - lineOpacity) * 255 / 100)
        fillOpacity = self.fillOpacityWidget.value()
        fillAlpha = int((100 - fillOpacity) * 255 / 100)
        vertexOpacity = self.pointOpacityWidget.value()
        vertexAlpha = int((100 - vertexOpacity) * 255 / 100)
        shape.line_color = QtGui.QColor(r, g, b, lineAlpha)
        shape.vertex_fill_color = QtGui.QColor(r, g, b, vertexAlpha)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255, 100)  # 70% transparency
        shape.fill_color = QtGui.QColor(r, g, b, fillAlpha)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, min(fillAlpha + 50, 255))

    def _get_rgb_by_label(self, label: str) -> tuple[int, int, int]:
        if self._config["shape_color"] == "auto":
            # Use label's position in the sorted label list for consistent colors
            label_id = self.uniqLabelList.get_label_index(label)
            if label_id is None:
                # Label not in list yet, use next available index
                label_id = self.uniqLabelList.count()
            label_id = (
                label_id + self._config["shift_auto_shape_color"]
            ) % len(LABEL_COLORMAP)
            rgb: tuple[int, int, int] = tuple(
                LABEL_COLORMAP[label_id].tolist()
            )
            return rgb
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            if not (
                len(self._config["label_colors"][label]) == 3
                and all(0 <= c <= 255 for c in self._config["label_colors"][label])
            ):
                raise ValueError(
                    "Color for label must be 0-255 RGB tuple, but got: "
                    f"{self._config['label_colors'][label]}"
                )
            return tuple(self._config["label_colors"][label])
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def _refresh_all_shape_colors(self):
        """Refresh colors of all shapes based on current uniqLabelList indices."""
        for row in range(self.labelList._model.rowCount()):
            item = self.labelList._model.item(row)
            if item:
                shape = item.shape()
                if shape:
                    self._update_shape_color(shape)
                    # Update label list item text
                    if shape.group_id is None:
                        text = shape.label
                    else:
                        text = f"{shape.label} ({shape.group_id})"
                    if shape.points:
                        x = min(p.x() for p in shape.points)
                        y = min(p.y() for p in shape.points)
                    else:
                        x, y = 0, 0
                    r, g, b = shape.fill_color.getRgb()[:3]
                    item.setText(
                        f'{html.escape(text)} <font color="#{r:02x}{g:02x}{b:02x}">●</font> ({int(x)},{int(y)})'
                    )

    def remLabels(self, shapes):
        # An explicit delete is not a drag/drop reorder, so the reorder
        # notification is suppressed - but only that one. Blocking the model's
        # own signals would leave the label list's shape index pointing at
        # deleted rows.
        by_shape = self.labelList.itemsByShape()
        items = [by_shape.get(shape) for shape in shapes]
        was_syncing = self._syncing_label_selection
        self._syncing_label_selection = True
        try:
            self.labelList.removeItems(items, emit_item_dropped=False)
        finally:
            self._syncing_label_selection = was_syncing

    def _load_shapes(self, shapes: list[Shape], replace: bool = True) -> None:
        self.labelList.itemSelectionChanged.disconnect(self._label_selection_changed)
        shape: Shape
        if not self._addLabelsSorted(shapes):
            for shape in shapes:
                self.addLabel(shape)
        # Re-update all shape colors after all labels are added to uniqLabelList
        # This is necessary because sorted insertion may change label indices
        self._refresh_all_shape_colors()
        self.labelList.clearSelection()
        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        self.canvas.loadShapes(shapes=shapes, replace=replace)
        self._recompute_reference_medians()

    def _load_shape_dicts(self, shape_dicts: list[ShapeDict]) -> None:
        shapes: list[Shape] = []
        shape_dict: ShapeDict
        for shape_dict in shape_dicts:
            shape: Shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                group_id=shape_dict["group_id"],
                description=shape_dict["description"],
                mask=shape_dict["mask"],
            )
            points = shape_dict["points"]
            # Normalize rectangles with 4 points to 2 diagonal corner points
            if shape_dict["shape_type"] == "rectangle" and len(points) == 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if not isinstance(shape.label, str):
                        logger.warning("shape.label is not str: {}", shape.label)
                        continue
                    if re.match(pattern, shape.label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(shape_dict["flags"])
            shape.other_data = shape_dict["other_data"]
            # Restore modified_at (check both direct field and other_data)
            # If not present, leave as None (will show red overlay in update distribution)
            if "modified_at" in shape_dict:
                shape.modified_at = shape_dict["modified_at"]
            elif "modified_at" in shape_dict["other_data"]:
                shape.modified_at = shape_dict["other_data"]["modified_at"]
            else:
                shape.modified_at = None

            shapes.append(shape)
        self._load_shapes(shapes=shapes)

    def _load_flags(self, flags: dict[str, bool]) -> None:
        self.flag_widget.clear()  # type: ignore[union-attr]
        key: str
        flag: bool
        for key, flag in flags.items():
            item: QtWidgets.QListWidgetItem = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                    modified_at=s.modified_at,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            assert item
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            assert self.imagePath
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self._other_data,
                flags=flags,
            )
            self.labelFile = lf
            # Update the file list item using stored row index
            if self._current_file_row >= 0:
                item = self.fileListWidget.item(self._current_file_row)
                if item:
                    path = item.data(Qt.UserRole) or item.text()
                    try:
                        self._file_mtimes[path] = osp.getmtime(filename)
                    except OSError:
                        pass
                    self._setFileItemAnnotated(item, True, saved_in_session=True)
                    self._refresh_file_list_colors()
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        # Create copies with offset (in image coordinates, adjusted for zoom)
        screen_offset = 30  # Constant offset in screen pixels
        image_offset = screen_offset / self.canvas.scale  # Convert to image coordinates
        new_shapes = []
        for shape in self._copied_shapes:
            new_shape = shape.copy()
            for point in new_shape.points:
                point.setX(point.x() + image_offset)
                point.setY(point.y() + image_offset)
            new_shapes.append(new_shape)
        # Update _copied_shapes with offset for next paste
        self._copied_shapes = new_shapes
        # Create shapes to load (copies to avoid reusing same objects)
        shapes_to_load = [s.copy() for s in new_shapes]
        self._load_shapes(shapes=shapes_to_load, replace=False)
        # Select only the newly pasted shapes
        self.canvas.selectShapes(shapes_to_load)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def _loadShapesFromJsonFile(self, label_file: str) -> bool:
        """Load shapes from a JSON label file and append to current canvas.

        Returns True if shapes were loaded successfully.
        """
        try:
            lf = LabelFile(label_file)
        except LabelFileError:
            return False
        if not lf.shapes:
            return False
        shapes: list[Shape] = []
        for sd in lf.shapes:
            shape = Shape(
                label=sd["label"],
                shape_type=sd["shape_type"],
                group_id=sd["group_id"],
                description=sd.get("description", ""),
                mask=sd.get("mask"),
            )
            points = sd["points"]
            if sd["shape_type"] == "rectangle" and len(points) == 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                points = [[min(xs), min(ys)], [max(xs), max(ys)]]
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()
            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if isinstance(shape.label, str) and re.match(
                        pattern, shape.label
                    ):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(sd.get("flags", {}))
            shape.other_data = sd.get("other_data", {})
            shape.modified_at = None
            shapes.append(shape)
        self._load_shapes(shapes=shapes, replace=False)
        self.setDirty()
        return True

    def addBase64ToAllJsons(self) -> None:
        """Iterate over all image entries in the file list and write base64
        image data into each corresponding JSON.
        """
        if not self.imageList:
            return
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "ファイル一覧の全画像に対応するJSONを開き、\n"
            "base64の画像データを書き込みます。\n"
            "（既に書かれている場合は上書きされます）\n\n"
            "本当に実施しますか？"
        )
        answer = mb.question(
            self,
            self.tr("確認"),
            msg,
            mb.Yes | mb.No,
            mb.No,
        )
        if answer != mb.Yes:
            return
        self._batch_update_jsons_image_data(add=True)

    def removeBase64FromAllJsons(self) -> None:
        """Iterate over all image entries in the file list and clear the
        base64 image data from each corresponding JSON.
        """
        if not self.imageList:
            return
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "ファイル一覧の全画像に対応するJSONを開き、\n"
            "base64の画像データを削除します。\n\n"
            "本当に実施しますか？"
        )
        answer = mb.question(
            self,
            self.tr("確認"),
            msg,
            mb.Yes | mb.No,
            mb.No,
        )
        if answer != mb.Yes:
            return
        self._batch_update_jsons_image_data(add=False)

    def _batch_update_jsons_image_data(self, add: bool) -> None:
        image_paths = self.imageList
        total = len(image_paths)
        title = (
            self.tr("全JSONに画像データ追加")
            if add
            else self.tr("全JSONから画像データ削除")
        )
        progress = QtWidgets.QProgressDialog(
            self.tr("JSON更新中..."),
            self.tr("キャンセル"),
            0,
            total,
            self,
        )
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        updated = 0
        unchanged = 0
        skipped = 0
        errors: list[str] = []
        # output_dir 利用時、複数の画像が同じ basename を持つと同一 JSON に
        # 衝突するため、処理済み label_file を覚えて重複処理を防ぐ。
        processed_label_files: set[str] = set()
        for idx, img_path in enumerate(image_paths):
            if progress.wasCanceled():
                break
            if (idx & 0x7) == 0:
                progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
            label_file = self._labelFileForImage(img_path)
            if label_file in processed_label_files:
                skipped += 1
                continue
            processed_label_files.add(label_file)
            if not osp.exists(label_file):
                skipped += 1
                continue
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if add:
                    if not osp.exists(img_path):
                        skipped += 1
                        continue
                    image_bytes = LabelFile.load_image_file(img_path)
                    if image_bytes is None:
                        skipped += 1
                        continue
                    new_value = base64.b64encode(image_bytes).decode("utf-8")
                    if data.get("imageData") == new_value:
                        unchanged += 1
                        continue
                    data["imageData"] = new_value
                else:
                    if data.get("imageData") is None:
                        unchanged += 1
                        continue
                    data["imageData"] = None
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                try:
                    self._file_mtimes[img_path] = osp.getmtime(label_file)
                except OSError:
                    pass
                updated += 1
            except Exception as e:
                errors.append(f"{osp.basename(label_file)}: {e}")
        progress.setValue(total)
        self._refresh_file_list_colors()
        summary = self.tr(
            "更新: {}件 / 変更なし: {}件 / スキップ: {}件 / エラー: {}件"
        ).format(updated, unchanged, skipped, len(errors))
        if errors:
            details = "\n".join(errors[:20])
            if len(errors) > 20:
                details += "\n..."
            QtWidgets.QMessageBox.warning(
                self,
                title,
                f"{summary}\n\n{details}",
            )
        else:
            QtWidgets.QMessageBox.information(self, title, summary)

    def removeHighIouShapes(self):
        """Remove duplicate shapes whose IoU exceeds 0.9, keeping the older one."""
        shapes = self.canvas.shapes
        if len(shapes) < 2:
            return

        def _bbox(shape):
            rect = shape.boundingRect()
            return rect.x(), rect.y(), rect.right(), rect.bottom()

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        # boundingRect() builds a QPainterPath per call; hoist it out of the
        # O(n^2) pair loop (shapes are not mutated until after confirmation).
        bboxes = [_bbox(s) for s in shapes]

        # Only shapes with the same label and type can be duplicates, and an
        # IoU above 0.9 requires overlapping x ranges, so compare inside each
        # group in x order and stop as soon as the ranges no longer touch.
        # This turns the full O(n^2) sweep into a near-linear one.
        groups: dict[tuple, list[int]] = {}
        for idx, shape in enumerate(shapes):
            groups.setdefault((shape.label, shape.shape_type), []).append(idx)

        to_remove: set[int] = set()
        for members in groups.values():
            members.sort(key=lambda idx: bboxes[idx][0])
            for a, i in enumerate(members):
                if i in to_remove:
                    continue
                for j in members[a + 1:]:
                    if bboxes[j][0] > bboxes[i][2]:
                        break  # sorted by left edge: no later shape can overlap
                    if j in to_remove:
                        continue
                    if _iou(bboxes[i], bboxes[j]) > 0.9:
                        # Keep the one with modified_at set; if both set, keep newer
                        ts_i = shapes[i].modified_at
                        ts_j = shapes[j].modified_at
                        if ts_i and not ts_j:
                            # i has timestamp, j doesn't → remove j
                            to_remove.add(j)
                        elif ts_j and not ts_i:
                            # j has timestamp, i doesn't → remove i
                            to_remove.add(i)
                            break
                        elif ts_i and ts_j:
                            # Both have timestamps → remove older
                            if ts_i < ts_j:
                                to_remove.add(i)
                                break
                            else:
                                to_remove.add(j)
                        else:
                            # Neither has timestamp → remove either
                            to_remove.add(j)

        if not to_remove:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("IoU0.9超シェイプを除去"),
                self.tr("IoU0.9超の重複シェイプはありません。"),
            )
            return

        count = len(to_remove)
        yes = QtWidgets.QMessageBox.Yes
        no = QtWidgets.QMessageBox.No
        result = QtWidgets.QMessageBox.question(
            self,
            self.tr("IoU0.9超シェイプを除去"),
            self.tr("IoU0.9超の重複シェイプが%d個あります。除去しますか？") % count,
            yes | no,
            yes,
        )
        if result != yes:
            return

        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            removed_shapes = [shapes[idx] for idx in sorted(to_remove, reverse=True)]
            # Remove them as a single transaction: deleteShape() snapshots
            # after every shape, so a bulk removal used to push the earlier
            # states out of the (10 deep) undo stack and become unrecoverable.
            remaining = [s for s in self.canvas.shapes if s not in set(removed_shapes)]
            self.canvas.shapes = remaining
            self.canvas.selectedShapes = [
                s for s in self.canvas.selectedShapes if s in set(remaining)
            ]
            self.canvas.sortShapesByArea()
            self.canvas._clearStaleHoverState()
            self.canvas.storeShapes()
            self.canvas.update()
            self.remLabels(removed_shapes)
            self.setDirty()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(
            self,
            self.tr("IoU0.9超シェイプを除去"),
            self.tr("%d個のシェイプを除去しました。") % count,
        )

    def copyShapesFromSpecifiedJson(self):
        """Open a file dialog to select a JSON file and copy all its shapes."""
        caption = self.tr("シェイプ複製元のJSONファイルを選択")
        start_dir = self.currentPath() if self.filename else "."
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption,
            start_dir,
            self.tr("Label files (*.json)"),
        )
        if not filename:
            return
        if not self._loadShapesFromJsonFile(filename):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("エラー"),
                self.tr("シェイプを読み込めませんでした。"),
            )

    def copyShapesFromPreviousJson(self):
        """Copy all shapes from the nearest previous file that has a saved JSON."""
        current_row = self.fileListWidget.currentRow()
        if current_row < 0:
            return

        for i in range(current_row - 1, -1, -1):
            item = self.fileListWidget.item(i)
            prev_path = item.data(Qt.UserRole) or item.text()
            label_file = self._labelFileForImage(prev_path)
            if not osp.exists(label_file):
                continue
            if self._loadShapesFromJsonFile(label_file):
                return

        QtWidgets.QMessageBox.warning(
            self,
            self.tr("エラー"),
            self.tr("保存されたJSONが存在しません。"),
        )

    def _label_selection_changed(self) -> None:
        if self._syncing_label_selection:
            return  # our own selection sync, not the user picking a row
        selected_shapes: list[Shape] = []
        for item in self.labelList.selectedItems():
            selected_shapes.append(item.shape())
        if selected_shapes:
            self.canvas.selectShapes(selected_shapes)
            # Center viewport on selected shape if it's outside the visible area
            self._center_on_shape(selected_shapes[0])
        else:
            self.canvas.deSelectShape()

    def _center_on_shape(self, shape: Shape) -> None:
        """Center the viewport on the given shape if it's outside visible area."""
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            return
        if not shape.points:
            return

        # Calculate shape center
        xs = [p.x() for p in shape.points]
        ys = [p.y() for p in shape.points]
        shape_center_x = (min(xs) + max(xs)) / 2
        shape_center_y = (min(ys) + max(ys)) / 2

        img_w = self.canvas.pixmap.width()
        img_h = self.canvas.pixmap.height()
        scale = self.canvas.scale

        view_w = self.scrollArea.viewport().width() / scale
        view_h = self.scrollArea.viewport().height() / scale

        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # Calculate current visible area
        offset = self.canvas.offsetToCenter()
        current_x = (h_bar.value() / scale) - offset.x()
        current_y = (v_bar.value() / scale) - offset.y()

        # Check if shape center is within visible area
        visible_left = max(0, current_x)
        visible_top = max(0, current_y)
        visible_right = min(img_w, current_x + view_w)
        visible_bottom = min(img_h, current_y + view_h)

        # Add some margin (10% of view size)
        margin_x = view_w * 0.1
        margin_y = view_h * 0.1

        is_visible = (
            visible_left + margin_x <= shape_center_x <= visible_right - margin_x
            and visible_top + margin_y <= shape_center_y <= visible_bottom - margin_y
        )

        if is_visible:
            return  # Shape is already visible, no need to scroll

        # Center on shape using ratio-based navigation
        x_ratio = shape_center_x / img_w if img_w > 0 else 0.5
        y_ratio = shape_center_y / img_h if img_h > 0 else 0.5
        self._onNavigatorViewportChange(x_ratio, y_ratio)

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)

        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            # Update last used label for changeSame action
            self._last_label = text
            self._update_change_same_action()
            self._update_pending_draw_label()
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

        # Refresh cursor overlay after label dialog closes
        self.canvas.refreshCursorOverlay()
        self._recompute_reference_medians()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.03  # natural scroll (reduced for Wacom compatibility)
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        bar = self.scrollBars[orientation]
        old_value = bar.value()
        bar.setValue(int(value))
        if self.filename is not None:
            # Persist the value actually applied after Qt range clamping.
            self.scroll_values[orientation][self.filename] = bar.value()
        if bar.value() == old_value:
            # setValue did not fire valueChanged, but other inputs of the
            # viewport computation may still have changed (e.g. FIT_WIDTH +
            # height-only window resize keeps zoom and scroll identical while
            # the viewport height changes), so update explicitly. When the
            # value did change, valueChanged already triggered this once.
            self._updateNavigatorViewport()

    def _set_zoom(self, value: int, pos: QtCore.QPointF | None = None) -> None:
        if self.filename is None:
            logger.warning("filename is None, cannot set zoom")
            return

        if pos is None:
            pos = QtCore.QPointF(self.canvas.visibleRegion().boundingRect().center())
        # Capture pre-zoom canvas mapping. We must record cursor → image
        # coordinates and the cursor's viewport-relative position BEFORE the
        # canvas resizes, because (1) offsetToCenter() can change
        # discontinuously when the 1.167x drag buffer toggles or when the
        # canvas transitions between viewport-fitted and image-fitted sizes,
        # and (2) Qt may clamp scroll bar values during the resize, making
        # post-resize scroll reads unreliable.
        scale_old: float = self.canvas.scale
        offset_old: QtCore.QPointF = self.canvas.offsetToCenter()
        scroll_x_old: int = self.scrollBars[Qt.Horizontal].value()
        scroll_y_old: int = self.scrollBars[Qt.Vertical].value()
        image_x: float = pos.x() / scale_old - offset_old.x()
        image_y: float = pos.y() / scale_old - offset_old.y()
        viewport_cursor_x: float = pos.x() - scroll_x_old
        viewport_cursor_y: float = pos.y() - scroll_y_old

        self.actions.fitWidth.setChecked(self._zoom_mode == _ZoomMode.FIT_WIDTH)
        self.actions.fitWindow.setChecked(self._zoom_mode == _ZoomMode.FIT_WINDOW)
        self._sync_fit_actions()
        self.canvas.enableDragging(
            enabled=value > int(self.scalers[_ZoomMode.FIT_WINDOW]() * 100)
        )
        self.zoomWidget.setValue(value)  # triggers self._paint_canvas
        self._zoom_values[self.filename] = (self._zoom_mode, value)

        scale_new: float = self.canvas.scale
        offset_new: QtCore.QPointF = self.canvas.offsetToCenter()
        new_canvas_x: float = (image_x + offset_new.x()) * scale_new
        new_canvas_y: float = (image_y + offset_new.y()) * scale_new
        # Absolute scroll values keep the same image point under the cursor.
        new_scroll_x: float = new_canvas_x - viewport_cursor_x
        new_scroll_y: float = new_canvas_y - viewport_cursor_y

        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]
        can_scroll_x = h_bar.maximum() > h_bar.minimum()
        can_scroll_y = v_bar.maximum() > v_bar.minimum()

        # In low-zoom center-lock mode, both ranges are zero; cursor anchoring
        # cannot be represented via scroll values, so skip adjustment.
        if not can_scroll_x and not can_scroll_y:
            return

        if can_scroll_x:
            self.setScroll(Qt.Horizontal, new_scroll_x)
        if can_scroll_y:
            self.setScroll(Qt.Vertical, new_scroll_y)

    def _set_zoom_to_original(self):
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=100)

    def _add_zoom(self, increment: float, pos: QtCore.QPointF | None = None) -> None:
        """Zoom to the next/previous predefined zoom level."""
        if pos is None:
            cursor_pos = self.canvas.mapFromGlobal(QtGui.QCursor.pos())
            if self.canvas.rect().contains(cursor_pos):
                pos = QtCore.QPointF(cursor_pos)
        current = self.zoomWidget.value()
        if increment > 1:
            # Zoom in: find the next larger level
            for level in self.ZOOM_LEVELS:
                if level > current:
                    zoom_value = level
                    break
            else:
                zoom_value = self.ZOOM_LEVELS[-1]  # Max level
        else:
            # Zoom out: find the next smaller level
            for level in reversed(self.ZOOM_LEVELS):
                if level < current:
                    zoom_value = level
                    break
            else:
                zoom_value = self.ZOOM_LEVELS[0]  # Min level
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=zoom_value, pos=pos)

    def _zoom_requested(self, delta: int, pos: QtCore.QPointF) -> None:
        scale_factor = 1 + delta / 1200
        self._linear_zoom(scale_factor, pos)

    def _pinch_zoom_requested(
        self, scale_factor: float, pos: QtCore.QPointF
    ) -> None:
        self._linear_zoom(scale_factor, pos=None)

    def _linear_zoom(
        self, scale_factor: float, pos: QtCore.QPointF | None = None
    ) -> None:
        """Zoom linearly by multiplying the current zoom by scale_factor."""
        current = self.zoomWidget.value()
        new_value = int(round(current * scale_factor))
        new_value = max(self.ZOOM_LEVELS[0], min(self.ZOOM_LEVELS[-1], new_value))
        if new_value == current:
            # Ensure at least 1% change
            new_value = current + (1 if scale_factor > 1 else -1)
            new_value = max(self.ZOOM_LEVELS[0], min(self.ZOOM_LEVELS[-1], new_value))
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=new_value, pos=pos)

    def _updateNavigatorViewport(self):
        """Update the navigator's viewport rectangle based on current scroll/zoom."""
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            return

        # Get image dimensions
        img_w = self.canvas.pixmap.width()
        img_h = self.canvas.pixmap.height()
        if img_w == 0 or img_h == 0:
            return

        scale = self.canvas.scale

        # Get scrollbar positions
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # Canvas centers the image using offsetToCenter()
        offset = self.canvas.offsetToCenter()

        # Scroll position in widget coords -> image coords, accounting for centering offset
        x = (h_bar.value() / scale) - offset.x()
        y = (v_bar.value() / scale) - offset.y()

        # Visible area in image coordinates
        view_w = self.scrollArea.viewport().width() / scale
        view_h = self.scrollArea.viewport().height() / scale

        # Calculate the visible rectangle clipped to image bounds
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(img_w), x + view_w)
        y2 = min(float(img_h), y + view_h)

        # Clipped width and height
        clipped_w = max(0.0, x2 - x1)
        clipped_h = max(0.0, y2 - y1)

        # Convert to ratios (0-1)
        x_ratio = x1 / img_w
        y_ratio = y1 / img_h
        w_ratio = clipped_w / img_w
        h_ratio = clipped_h / img_h

        self.navigator.setViewportRect(x_ratio, y_ratio, w_ratio, h_ratio)
        self.update_distribution.setViewportRect(x_ratio, y_ratio, w_ratio, h_ratio)
        self._refreshDockShapes()

    def _onNavigatorViewportChange(self, x_ratio: float, y_ratio: float):
        """Handle click on navigator to move viewport center."""
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            return

        img_w = self.canvas.pixmap.width()
        img_h = self.canvas.pixmap.height()
        scale = self.canvas.scale

        view_w = self.scrollArea.viewport().width() / scale
        view_h = self.scrollArea.viewport().height() / scale

        # Calculate target top-left position in image coords (center on clicked point)
        target_x = x_ratio * img_w - view_w / 2
        target_y = y_ratio * img_h - view_h / 2

        # Account for canvas centering offset
        offset = self.canvas.offsetToCenter()

        # Convert to scrollbar values (reverse of _updateNavigatorViewport calculation)
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        h_value = (target_x + offset.x()) * scale
        v_value = (target_y + offset.y()) * scale

        # Clamp to valid range
        h_value = max(0, min(h_bar.maximum(), h_value))
        v_value = max(0, min(v_bar.maximum(), v_value))

        self.setScroll(Qt.Horizontal, h_value)
        self.setScroll(Qt.Vertical, v_value)

    def _line_opacity_changed(self, value: int) -> None:
        """Update line opacity for all shapes."""
        # value is transparency (100 = fully transparent, 0 = fully opaque)
        alpha = int((100 - value) * 255 / 100)
        # Update class-level default line color
        Shape.line_color.setAlpha(alpha)
        # Update existing shapes (only if canvas exists and they have instance-level line_color)
        if hasattr(self, "canvas") and self.canvas is not None:
            for shape in self.canvas.shapes:
                if "line_color" in shape.__dict__:
                    shape.line_color.setAlpha(alpha)
            self.canvas.update()

    def _vertex_opacity_changed(self, value: int) -> None:
        """Update opacity for vertices (polygon corners)."""
        # value is transparency (100 = fully transparent, 0 = fully opaque)
        alpha = int((100 - value) * 255 / 100)
        # Update vertex_fill_color for all shapes (hvertex_fill_color is fixed at 90%)
        if hasattr(self, "canvas") and self.canvas is not None:
            for shape in self.canvas.shapes:
                if "vertex_fill_color" in shape.__dict__:
                    shape.vertex_fill_color.setAlpha(alpha)
            self.canvas.update()

    def _fill_opacity_changed(self, value: int) -> None:
        """Update fill opacity for all shapes."""
        # value is transparency (100 = fully transparent, 0 = fully opaque)
        alpha = int((100 - value) * 255 / 100)
        # Update class-level default fill color
        Shape.fill_color.setAlpha(alpha)
        Shape.select_fill_color.setAlpha(min(alpha + 50, 255))
        # Update existing shapes (only if canvas exists and they have instance-level fill_color)
        if hasattr(self, "canvas") and self.canvas is not None:
            for shape in self.canvas.shapes:
                if "fill_color" in shape.__dict__:
                    shape.fill_color.setAlpha(alpha)
                if "select_fill_color" in shape.__dict__:
                    shape.select_fill_color.setAlpha(min(alpha + 50, 255))
            self.canvas.update()

    def _parallel_line_dist_toggled(self, checked: bool) -> None:
        if checked and hasattr(self, "lineFitCheckbox") and self.lineFitCheckbox.isChecked():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("排他機能"),
                self.tr("「直線へのフィット」と同時に有効化はできません。"),
            )
            self.parallelLineDistCheckbox.setChecked(False)
            return
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setParallelLineDistEnabled(checked)

    def _magnet_rules(self, key: str) -> list[dict]:
        """Rules for one magnet with `magnet_defaults` filled in.

        Keys shared by every magnet (absolute_pixels, resize_base, ...) can
        be written once at the top level instead of repeating them in each
        rule; a value set on the rule itself always wins.
        """
        defaults = self._config.get("magnet_defaults") or {}
        rules = self._config.get(key) or []
        if not defaults:
            return rules
        return [{**defaults, **rule} for rule in rules]

    def _recompute_reference_medians(self) -> None:
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        medians: dict[str, float | None] = {}
        # Parallel line rules — margin from resize_base / margin_pixels
        for i, rule in enumerate(self._magnet_rules("parallel_line_magnet")):
            margin_px = rule.get("margin_pixels", 10)
            if self.canvas.pixmap is not None and not rule.get("absolute_pixels"):
                resize_base = rule.get("resize_base", 2560)
                img_w = self.canvas.pixmap.width()
                img_h = self.canvas.pixmap.height()
                scale = max(img_w, img_h) / resize_base
                medians[f"pl:{i}"] = margin_px * scale
            else:
                # absolute_pixels (or no image yet): plain image pixels
                medians[f"pl:{i}"] = float(margin_px)
        # Text bounding: margin/snap computed directly in canvas from rule params
        self.canvas.setReferenceMedians(medians)

    def _toggle_parallel_line_dist_visible(self, checked: bool) -> None:
        if hasattr(self, "_parallelLineDistAction"):
            self._parallelLineDistAction.setVisible(checked)

    def _text_bounding_toggled(self, checked: bool) -> None:
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setTextBoundingEnabled(checked)

    def _toggle_text_bounding_visible(self, checked: bool) -> None:
        if hasattr(self, "_textBoundingAction"):
            self._textBoundingAction.setVisible(checked)

    def _line_fit_toggled(self, checked: bool) -> None:
        if checked and hasattr(self, "parallelLineDistCheckbox") and self.parallelLineDistCheckbox.isChecked():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("排他機能"),
                self.tr("「平行直線との距離調整」と同時に有効化はできません。"),
            )
            self.lineFitCheckbox.setChecked(False)
            return
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setLineFitEnabled(checked)

    def _toggle_line_fit_visible(self, checked: bool) -> None:
        if hasattr(self, "_lineFitAction"):
            self._lineFitAction.setVisible(checked)

    def _dark_pixel_magnet_toggled(self, checked: bool) -> None:
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setDarkPixelMagnetEnabled(checked)

    def _toggle_dark_pixel_magnet_visible(self, checked: bool) -> None:
        if hasattr(self, "_darkPixelMagnetAction"):
            self._darkPixelMagnetAction.setVisible(checked)

    def _toggle_create_button_visible(self, action_name: str, checked: bool) -> None:
        act = getattr(self.actions, action_name, None)
        if act is None:
            return
        act.setVisible(checked)
        for tb in (
            getattr(self, "_tools_toolbar", None),
            getattr(self, "_create_toolbar", None),
        ):
            if tb is None:
                continue
            buttons = getattr(tb, "_action_buttons", {})
            toolbar_action = buttons.get(act)
            if toolbar_action is not None:
                toolbar_action.setVisible(checked)

    def _toggle_create_polygon3_visible(self, checked: bool) -> None:
        self._toggle_create_button_visible("createPolygon3Mode", checked)

    def _toggle_create_circle_visible(self, checked: bool) -> None:
        self._toggle_create_button_visible("createCircleMode", checked)

    def _toggle_create_line_visible(self, checked: bool) -> None:
        self._toggle_create_button_visible("createLineMode", checked)

    def _toggle_create_linestrip_visible(self, checked: bool) -> None:
        self._toggle_create_button_visible("createLineStripMode", checked)

    def _toggle_create_ai_polygon_visible(self, checked: bool) -> None:
        self._toggle_create_button_visible("createAiPolygonMode", checked)

    def _toggle_create_ai_mask_visible(self, checked: bool) -> None:
        self._toggle_create_button_visible("createAiMaskMode", checked)

    def _toggle_create_magic_wand_visible(self, checked: bool) -> None:
        act = self._magic_wand_action
        act.setVisible(checked)
        for tb in (
            getattr(self, "_tools_toolbar", None),
            getattr(self, "_create_toolbar", None),
        ):
            if tb is None:
                continue
            buttons = getattr(tb, "_action_buttons", {})
            toolbar_action = buttons.get(act)
            if toolbar_action is not None:
                toolbar_action.setVisible(checked)

    def _toggle_redo_visible(self, checked: bool) -> None:
        self.actions.redo.setVisible(checked)
        if hasattr(self, "_tools_toolbar"):
            buttons = getattr(self._tools_toolbar, "_action_buttons", {})
            toolbar_action = buttons.get(self.actions.redo)
            if toolbar_action is not None:
                toolbar_action.setVisible(checked)

    def _toggle_toolbar_button_visible(self, action, checked: bool) -> None:
        """Toggle visibility of a toolbar button without affecting menu QAction."""
        if hasattr(self, "_tools_toolbar"):
            buttons = getattr(self._tools_toolbar, "_action_buttons", {})
            toolbar_action = buttons.get(action)
            if toolbar_action is not None:
                toolbar_action.setVisible(checked)

    def _toggle_rect_to_triangle_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(self.actions.rectToTriangle, checked)

    def _toggle_delete_all_shapes_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(
            self.actions.deleteAllShapes, checked
        )

    def _toggle_remove_high_iou_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(
            self.actions.removeHighIouShapes, checked
        )

    def _toggle_copy_from_specified_json_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(
            self.actions.copyFromSpecifiedJson, checked
        )

    def _toggle_copy_from_prev_json_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(
            self.actions.copyFromPrevJson, checked
        )

    def _toggle_add_base64_to_all_jsons_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(
            self.actions.addBase64ToAllJsons, checked
        )

    def _toggle_remove_base64_from_all_jsons_visible(self, checked: bool) -> None:
        self._toggle_toolbar_button_visible(
            self.actions.removeBase64FromAllJsons, checked
        )

    def _line_width_changed(self, value: int) -> None:
        """Update line width for all shapes."""
        Shape.PEN_WIDTH = value
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.update()

    def _point_object_size_changed(self, value: int) -> None:
        """Update size for point shape_type objects only."""
        Shape.point_object_size = value
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.update()

    def _vertex_size_changed(self, value: int) -> None:
        """Update vertex size for polygon/rectangle vertices."""
        Shape.point_size = value
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.update()

    def _toggle_line_opacity_visible(self, checked: bool) -> None:
        if hasattr(self, "_lineOpacityAction"):
            self._lineOpacityAction.setVisible(checked)

    def _toggle_point_opacity_visible(self, checked: bool) -> None:
        if hasattr(self, "_pointOpacityAction"):
            self._pointOpacityAction.setVisible(checked)

    def _toggle_fill_opacity_visible(self, checked: bool) -> None:
        if hasattr(self, "_fillOpacityAction"):
            self._fillOpacityAction.setVisible(checked)

    def _toggle_line_width_visible(self, checked: bool) -> None:
        if hasattr(self, "_lineWidthAction"):
            self._lineWidthAction.setVisible(checked)

    def _toggle_point_object_size_visible(self, checked: bool) -> None:
        if hasattr(self, "_pointObjectSizeAction"):
            self._pointObjectSizeAction.setVisible(checked)

    def _toggle_vertex_size_visible(self, checked: bool) -> None:
        if hasattr(self, "_vertexSizeAction"):
            self._vertexSizeAction.setVisible(checked)

    def _store_image_data_toggled(self, checked: bool) -> None:
        self._config["store_data"] = bool(checked)
        if hasattr(self, "actions") and hasattr(self.actions, "saveWithImageData"):
            self.actions.saveWithImageData.setChecked(bool(checked))

    def _toggle_store_image_data_visible(self, checked: bool) -> None:
        if hasattr(self, "_storeImageDataAction"):
            self._storeImageDataAction.setVisible(checked)

    def _custom_cursor_toggled(self, checked: bool) -> None:
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setCustomCursorEnabled(checked)

    def _right_click_edit_toggled(self, checked: bool) -> None:
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setRightClickEditEnabled(checked)

    def _auto_fit_toggled(self, checked: bool) -> None:
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setAutoFitEnabled(checked)

    def _toggle_auto_fit_visible(self, checked: bool) -> None:
        if hasattr(self, "_autoFitAction"):
            self._autoFitAction.setVisible(checked)

    def _sync_fit_actions(self) -> None:
        """Disable the fit button whose mode is already active.

        Re-clicking a checked fit button would toggle it off and drop the
        zoom to manual 100%, which surprises users; graying it out instead
        makes the active mode explicit.
        """
        self.actions.fitWindow.setEnabled(self._zoom_mode != _ZoomMode.FIT_WINDOW)
        self.actions.fitWidth.setEnabled(self._zoom_mode != _ZoomMode.FIT_WIDTH)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self._zoom_mode = _ZoomMode.FIT_WINDOW if value else _ZoomMode.MANUAL_ZOOM
        self._adjust_scale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self._zoom_mode = _ZoomMode.FIT_WIDTH if value else _ZoomMode.MANUAL_ZOOM
        self._adjust_scale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value: bool, is_initial_load: bool = False):
        if self.filename is None:
            logger.warning("filename is None, cannot set brightness/contrast")
            return

        if is_initial_load:
            brightness, contrast = self._brightness_contrast_values.get(
                self.filename, (None, None)
            )
            if self._config["keep_prev_brightness_contrast"] and self.recentFiles:
                brightness, contrast = self._brightness_contrast_values.get(
                    self.recentFiles[0], (None, None)
                )
            if brightness in (None, 50) and contrast in (None, 50):
                # Neutral values reproduce the already-loaded pixmap exactly,
                # so skip building the dialog: on a 4K image that pass costs a
                # full PIL decode plus a second Canvas.loadPixmap (~700 ms).
                self._brightness_contrast_values[self.filename] = (
                    brightness,
                    contrast,
                )
                return

        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData).convert("RGB"),
            self.onNewBrightnessContrast,
            parent=self,
        )

        brightness: int | None
        contrast: int | None
        brightness, contrast = self._brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if is_initial_load:
            prev_filename: str = self.recentFiles[0] if self.recentFiles else ""
            if self._config["keep_prev_brightness_contrast"] and prev_filename:
                brightness, contrast = self._brightness_contrast_values.get(
                    prev_filename, (None, None)
                )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)

        if is_initial_load:
            dialog.onNewValue(None)
        else:
            dialog.exec_()
            brightness = dialog.slider_brightness.value()
            contrast = dialog.slider_contrast.value()

        self._brightness_contrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def _load_file(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        prev_shapes: list[Shape] = []
        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.show_status_message(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = self._labelFileForImage(filename)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p><p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.show_status_message(self.tr("Error reading %s") % label_file)
                return False
            assert self.labelFile is not None
            self.imageData = self.labelFile.imageData
            assert self.labelFile.imagePath
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self._other_data = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        assert self.imageData is not None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                f"*.{fmt.data().decode()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.show_status_message(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        pixmap = QtGui.QPixmap.fromImage(image)
        self.canvas.loadPixmap(pixmap)
        self.navigator.setPixmap(pixmap)
        self.update_distribution.setPixmap(pixmap)
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self._load_shape_dicts(shape_dicts=self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        self._load_flags(flags=flags)
        if prev_shapes and self.noShapes():
            self._load_shapes(shapes=prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self._zoom_values
        if self.filename in self._zoom_values:
            self._zoom_mode = self._zoom_values[self.filename][0]
            self._set_zoom(self._zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self._zoom_mode = _ZoomMode.FIT_WINDOW
            self._adjust_scale()
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        self.brightnessContrast(value=False, is_initial_load=True)
        # Apply current opacity and line width settings to loaded shapes
        self._line_opacity_changed(self.lineOpacityWidget.value())
        self._vertex_opacity_changed(self.pointOpacityWidget.value())
        self._fill_opacity_changed(self.fillOpacityWidget.value())
        self._line_width_changed(self.lineWidthWidget.value())
        self._point_object_size_changed(self.pointObjectSizeWidget.value())
        self._vertex_size_changed(self.vertexSizeWidget.value())
        self._paint_canvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        # Store the row index of the currently loaded file
        self._current_file_row = self.fileListWidget.currentRow()
        self.show_status_message(self.tr("Loaded %s") % osp.basename(filename))
        # Pre-build the image caches once the window is responsive again:
        # keeps file switching fast while avoiding a freeze on the first
        # mouse move (which needs them for the pixel readout and magnets).
        QtCore.QTimer.singleShot(0, self.canvas.warmImageCaches)
        logger.debug("loaded file: {!r}", filename)
        return True

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if (
            self.canvas
            and not self.image.isNull()
            and self._zoom_mode != _ZoomMode.MANUAL_ZOOM
        ):
            self._adjust_scale()
        super().resizeEvent(a0)

    def _paint_canvas(self) -> None:
        if self.image.isNull():
            logger.warning("image is null, cannot paint canvas")
            return
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()
        self._updateNavigatorViewport()

    def _adjust_scale(self) -> None:
        self._set_zoom(value=int(self.scalers[self._zoom_mode]() * 100))

    def scaleFitWindow(self) -> float:
        EPSILON_TO_HIDE_SCROLLBAR: float = 2.0
        w1: float = self.centralWidget().width() - EPSILON_TO_HIDE_SCROLLBAR
        h1: float = self.centralWidget().height() - EPSILON_TO_HIDE_SCROLLBAR
        a1: float = w1 / h1

        w2: float = self.canvas.pixmap.width()
        h2: float = self.canvas.pixmap.height()
        a2: float = w2 / h2

        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        EPSILON_TO_HIDE_SCROLLBAR: float = 15.0
        w = self.centralWidget().width() - EPSILON_TO_HIDE_SCROLLBAR
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)
        if hasattr(self, "storeImageDataCheckbox"):
            self.storeImageDataCheckbox.blockSignals(True)
            self.storeImageDataCheckbox.setChecked(enabled)
            self.storeImageDataCheckbox.blockSignals(False)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if not self._can_continue():
            a0.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue(
            "lastOpenedDir", self._prev_opened_dir if self._prev_opened_dir else ""
        )
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        self.settings.setValue("canvas/lineOpacity", self.lineOpacityWidget.value())
        self.settings.setValue("canvas/pointOpacity", self.pointOpacityWidget.value())
        self.settings.setValue("canvas/fillOpacity", self.fillOpacityWidget.value())
        self.settings.setValue("canvas/lineWidth", self.lineWidthWidget.value())
        self.settings.setValue(
            "canvas/pointObjectSize", self.pointObjectSizeWidget.value()
        )
        self.settings.setValue(
            "canvas/vertexSize", self.vertexSizeWidget.value()
        )
        for key, action_ in (
            ("view/showLineOpacity", "showLineOpacity"),
            ("view/showPointOpacity", "showPointOpacity"),
            ("view/showFillOpacity", "showFillOpacity"),
            ("view/showLineWidth", "showLineWidth"),
            ("view/showPointObjectSize", "showPointObjectSize"),
            ("view/showVertexSize", "showVertexSize"),
        ):
            self.settings.setValue(
                key, getattr(self.actions, action_).isChecked()
            )
        self.settings.setValue(
            "canvas/storeImageData", self.storeImageDataCheckbox.isChecked()
        )
        self.settings.setValue(
            "view/showStoreImageData",
            self.actions.showStoreImageData.isChecked(),
        )
        self.settings.setValue(
            "canvas/customCursor", self.customCursorCheckbox.isChecked()
        )
        self.settings.setValue(
            "canvas/rightClickEdit", self.rightClickEditCheckbox.isChecked()
        )
        self.settings.setValue(
            "canvas/skipDeleteConfirm",
            self.skipDeleteConfirmCheckbox.isChecked(),
        )
        self.settings.setValue(
            "canvas/skipSaveNameConfirm",
            self.skipSaveNameConfirmCheckbox.isChecked(),
        )
        self.settings.setValue(
            "canvas/parallelLineDist",
            self.parallelLineDistCheckbox.isChecked(),
        )
        self.settings.setValue(
            "view/showCreatePolygon3", self.actions.showCreatePolygon3.isChecked()
        )
        self.settings.setValue(
            "view/showCreateCircle", self.actions.showCreateCircle.isChecked()
        )
        self.settings.setValue(
            "view/showCreateLine", self.actions.showCreateLine.isChecked()
        )
        self.settings.setValue(
            "view/showCreateLineStrip",
            self.actions.showCreateLineStrip.isChecked(),
        )
        self.settings.setValue(
            "view/showCreateAiPolygon",
            self.actions.showCreateAiPolygon.isChecked(),
        )
        self.settings.setValue(
            "view/showCreateAiMask",
            self.actions.showCreateAiMask.isChecked(),
        )
        self.settings.setValue(
            "view/showCreateMagicWand",
            self.actions.showCreateMagicWand.isChecked(),
        )
        self.settings.setValue(
            "view/showRedo", self.actions.showRedo.isChecked()
        )
        for key, action_ in (
            ("view/showRectToTriangle", "showRectToTriangle"),
            ("view/showDeleteAllShapes", "showDeleteAllShapes"),
            ("view/showRemoveHighIouShapes", "showRemoveHighIouShapes"),
            ("view/showCopyFromSpecifiedJson", "showCopyFromSpecifiedJson"),
            ("view/showCopyFromPrevJson", "showCopyFromPrevJson"),
            ("view/showAddBase64ToAllJsons", "showAddBase64ToAllJsons"),
            ("view/showRemoveBase64FromAllJsons", "showRemoveBase64FromAllJsons"),
        ):
            self.settings.setValue(
                key, getattr(self.actions, action_).isChecked()
            )
        self.settings.setValue(
            "view/showParallelLineDist",
            self.actions.showParallelLineDist.isChecked(),
        )
        self.settings.setValue(
            "canvas/textBounding",
            self.textBoundingCheckbox.isChecked(),
        )
        self.settings.setValue(
            "view/showTextBounding",
            self.actions.showTextBounding.isChecked(),
        )
        self.settings.setValue(
            "canvas/lineFit",
            self.lineFitCheckbox.isChecked(),
        )
        self.settings.setValue(
            "view/showLineFit",
            self.actions.showLineFit.isChecked(),
        )
        self.settings.setValue(
            "canvas/darkPixelMagnet",
            self.darkPixelMagnetCheckbox.isChecked(),
        )
        self.settings.setValue(
            "view/showDarkPixelMagnet",
            self.actions.showDarkPixelMagnet.isChecked(),
        )
        self.settings.setValue(
            "canvas/autoFit",
            self.autoFitCheckbox.isChecked(),
        )
        self.settings.setValue(
            "view/showAutoFit",
            self.actions.showAutoFit.isChecked(),
        )
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if a0.mimeData().hasUrls():
            items = [i.toLocalFile() for i in a0.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                a0.accept()
        else:
            a0.ignore()

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        if not self._can_continue():
            a0.ignore()
            return
        items = [i.toLocalFile() for i in a0.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self._can_continue():
            self._load_file(filename)

    def _open_prev_image(self, _value=False) -> None:
        row_prev: int = self.fileListWidget.currentRow() - 1
        if row_prev < 0:
            logger.debug("there is no prev image")
            return

        logger.debug("setting current row to {:d}", row_prev)
        self.fileListWidget.setCurrentRow(row_prev)
        self.fileListWidget.repaint()
        self._update_nav_button_state()

    def _open_next_image(self, _value=False) -> None:
        row_next: int = self.fileListWidget.currentRow() + 1
        if row_next >= self.fileListWidget.count():
            logger.debug("there is no next image")
            return

        logger.debug("setting current row to {:d}", row_next)
        self.fileListWidget.setCurrentRow(row_next)
        self.fileListWidget.repaint()
        self._update_nav_button_state()

    def _update_nav_button_state(self) -> None:
        row = self.fileListWidget.currentRow()
        count = self.fileListWidget.count()
        self.actions.openPrevImg.setEnabled(row > 0)
        self.actions.openNextImg.setEnabled(row < count - 1)

    def _open_file_with_dialog(self, _value: bool = False) -> None:
        if not self._can_continue():
            return
        if self.filename:
            path = osp.dirname(str(self.filename))
        elif self._prev_opened_dir and osp.exists(self._prev_opened_dir):
            path = self._prev_opened_dir
        else:
            path = "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self._load_file(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self._import_images_from_dir(root_dir=self._prev_opened_dir)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False) -> bool:
        """Save the current annotations. Returns True only when written."""
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            return self._saveFile(self.labelFile.filename)
        elif self.output_file:
            saved = self._saveFile(self.output_file)
            self.close()
            return saved
        elif self.filename:
            if self.skipSaveNameConfirmCheckbox.isChecked():
                # Auto-derive JSON filename from image filename
                return self._saveFile(self._labelFileForImage(self.filename))
            return self._saveFile(self.saveFileDialog())
        else:
            # No filename set, cannot save
            logger.warning("Cannot save: no filename set")
            return False

    def saveFileAs(self, _value=False) -> bool:
        assert not self.image.isNull(), "cannot save empty image"
        return self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        assert self.filename is not None
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            return filename[0]
        return filename

    def _saveFile(self, filename) -> bool:
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()
            # Refresh cursor overlay after save
            self.canvas.refreshCursorOverlay()
            return True
        return False

    def closeFile(self, _value=False):
        if not self._can_continue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.fileListWidget.setFocus()
        self.actions.saveAs.setEnabled(False)

    def _labelFileForImage(self, image_path: str) -> str:
        """JSON path for an image, honouring output_dir.

        Every place that derives a label file must agree, otherwise saving,
        deleting and the "annotated" state look at different files.
        """
        if image_path.lower().endswith(LabelFile.suffix):
            label_file = image_path
        else:
            label_file = f"{osp.splitext(image_path)[0]}{LabelFile.suffix}"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))
        return label_file

    def getLabelFile(self):
        assert self.filename is not None
        return self._labelFileForImage(self.filename)

    def _compute_alpha_for_mtime(self, mtime: float) -> int:
        """Map mtime to alpha (FILE_ALPHA_MIN..FILE_ALPHA_MAX) using cached range."""
        rng = self._file_mtime_range
        if rng is None:
            return self.FILE_ALPHA_MAX
        min_mt, max_mt = rng
        if max_mt <= min_mt:
            return self.FILE_ALPHA_MAX
        t = (mtime - min_mt) / (max_mt - min_mt)
        t = max(0.0, min(1.0, t))
        return int(round(self.FILE_ALPHA_MIN + t * (self.FILE_ALPHA_MAX - self.FILE_ALPHA_MIN)))

    def _refresh_file_list_colors(self) -> None:
        """Recompute mtime range and re-apply gradient backgrounds to all items."""
        if not self._file_mtimes:
            self._file_mtime_range = None
        else:
            mts = self._file_mtimes.values()
            self._file_mtime_range = (min(mts), max(mts))
        r, g, b = self.FILE_COLOR_RGB
        self.fileListWidget.setUpdatesEnabled(False)
        try:
            for i in range(self.fileListWidget.count()):
                item = self.fileListWidget.item(i)
                if item is None:
                    continue
                path = item.data(Qt.UserRole) or item.text()
                mtime = self._file_mtimes.get(path)
                if mtime is None:
                    new_brush = QtGui.QBrush()
                else:
                    alpha = self._compute_alpha_for_mtime(mtime)
                    new_brush = QtGui.QBrush(QtGui.QColor(r, g, b, alpha))
                # QListWidgetItem.setBackground emits dataChanged even for an
                # equal brush; skip rows whose color is already correct.
                if item.background() != new_brush:
                    item.setBackground(new_brush)
        finally:
            self.fileListWidget.setUpdatesEnabled(True)

    def _setFileItemAnnotated(
        self, item: QtWidgets.QListWidgetItem, annotated: bool, saved_in_session: bool = False
    ) -> None:
        """Set file list item check state and background color.

        Background uses an mtime-based gradient (oldest=alpha10, newest=alpha60)
        when the file's mtime is in the cache. Falls back to the legacy max-alpha
        color when mtime is not yet known. ``saved_in_session`` is kept for API
        compatibility but no longer affects the rendering — mtime drives color.
        """
        if annotated:
            item.setCheckState(Qt.Checked)
            path = item.data(Qt.UserRole) or item.text()
            mtime = self._file_mtimes.get(path)
            if mtime is None:
                item.setBackground(self.FILE_NEWLY_SAVED_COLOR)
            else:
                alpha = self._compute_alpha_for_mtime(mtime)
                r, g, b = self.FILE_COLOR_RGB
                item.setBackground(QtGui.QColor(r, g, b, alpha))
        else:
            item.setCheckState(Qt.Unchecked)
            item.setBackground(QtGui.QBrush())  # Clear background

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info(f"Label file is removed: {label_file}")

            item = self.fileListWidget.currentItem()
            if item:
                path = item.data(Qt.UserRole) or item.text()
                self._file_mtimes.pop(path, None)
                self._setFileItemAnnotated(item, False)
                self._refresh_file_list_colors()

            self.resetState()

    def exportFileList(self):
        """Export file list to CSV with image info and annotation status."""
        if not self.imageList:
            return

        # Ask for save location
        default_name = "file_list.csv"
        if self._prev_opened_dir:
            default_path = osp.join(self._prev_opened_dir, default_name)
        else:
            default_path = default_name

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export File List"),
            default_path,
            self.tr("CSV files (*.csv)"),
        )
        if not filename:
            return

        # Collect data for each image
        rows = []
        for image_path in self.imageList:
            row = self._get_file_info(image_path)
            rows.append(row)

        # Write CSV
        headers = [
            "filename",
            "width",
            "height",
            "has_annotation",
            "annotation_modified",
            "num_shapes",
            "shape_types",
            "labels",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            logger.info(f"File list exported to: {filename}")
            self.show_status_message(self.tr("File list exported to %s") % filename)
        except Exception as e:
            self.errorMessage(
                self.tr("Export Error"),
                self.tr("Failed to export file list: %s") % str(e),
            )

    def _get_file_info(self, image_path: str) -> list:
        """Get information about an image file for CSV export."""
        filename = osp.basename(image_path)

        # Get image dimensions
        width, height = "", ""
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            pass

        # Check for annotation file
        label_file = self._labelFileForImage(image_path)

        has_annotation = "No"
        annotation_modified = ""
        num_shapes = 0
        shape_types = ""
        labels = ""

        if osp.exists(label_file):
            has_annotation = "Yes"
            try:
                mtime = osp.getmtime(label_file)
                annotation_modified = datetime.datetime.fromtimestamp(mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Read JSON to get shape info
                with open(label_file, encoding="utf-8") as f:
                    data = json.load(f)
                    shapes = data.get("shapes", [])
                    num_shapes = len(shapes)

                    # Count shape types
                    type_counts: dict[str, int] = {}
                    label_set: set[str] = set()
                    for shape in shapes:
                        stype = shape.get("shape_type", "unknown")
                        type_counts[stype] = type_counts.get(stype, 0) + 1
                        label_set.add(shape.get("label", ""))

                    shape_types = ", ".join(
                        f"{t}:{c}" for t, c in sorted(type_counts.items())
                    )
                    labels = ", ".join(sorted(label_set))
            except Exception:
                pass

        return [
            filename,
            width,
            height,
            has_annotation,
            annotation_modified,
            num_shapes,
            shape_types,
            labels,
        ]

    def progressStats(self):
        """Generate progress statistics chart as a 2x3 subplot image."""
        if not self.imageList:
            return

        import matplotlib.pyplot as plt
        from collections import Counter

        # Ask for save location
        default_name = "progress_stats.png"
        if self._prev_opened_dir:
            default_path = osp.join(self._prev_opened_dir, default_name)
        else:
            default_path = default_name

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save Progress Stats"),
            default_path,
            self.tr("PNG files (*.png);;All files (*)"),
        )
        if not filename:
            return

        # --- Collect data from all JSON files ---
        daily_counts: dict[str, int] = {}  # date_str -> count
        label_counter: Counter = Counter()
        shapes_per_file: list[int] = []
        vertices_per_file: list[int] = []
        area_per_file: list[float] = []

        for image_path in self.imageList:
            label_file = self._labelFileForImage(image_path)
            if not osp.exists(label_file):
                continue

            try:
                mtime = osp.getmtime(label_file)
                date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                daily_counts[date_str] = daily_counts.get(date_str, 0) + 1

                with open(label_file, encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data.get("shapes", [])
                shapes_per_file.append(len(shapes))

                # Get image dimensions for area ratio
                img_w = data.get("imageWidth", 0)
                img_h = data.get("imageHeight", 0)
                image_area = img_w * img_h if img_w and img_h else 0

                file_vertices = 0
                file_bbox_area = 0.0
                for shape in shapes:
                    label_counter[shape.get("label", "")] += 1
                    points = shape.get("points", [])
                    file_vertices += len(points)
                    # Bounding box area
                    if len(points) >= 2:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        file_bbox_area += (max(xs) - min(xs)) * (max(ys) - min(ys))
                vertices_per_file.append(file_vertices)
                # Convert to percentage of image area
                if image_area > 0:
                    area_per_file.append(file_bbox_area / image_area * 100)
                else:
                    area_per_file.append(0.0)
            except Exception:
                continue

        if not daily_counts:
            self.errorMessage(
                self.tr("No Data"),
                self.tr("No annotation files found to analyze."),
            )
            return

        # --- Sort dates and build time series ---
        sorted_dates = sorted(daily_counts.keys())
        # Fill missing dates with 0
        all_dates: list[str] = []
        all_counts: list[int] = []
        start = datetime.datetime.strptime(sorted_dates[0], "%Y-%m-%d")
        end = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
        current = start
        while current <= end:
            ds = current.strftime("%Y-%m-%d")
            all_dates.append(ds)
            all_counts.append(daily_counts.get(ds, 0))
            current += datetime.timedelta(days=1)

        cumsum = np.cumsum(all_counts)

        # Moving average (7-day)
        window = min(7, len(all_counts))
        if window > 0:
            kernel = np.ones(window) / window
            moving_avg = np.convolve(all_counts, kernel, mode="same")
        else:
            moving_avg = np.array(all_counts, dtype=float)

        # --- Create figure ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Annotation Progress Stats", fontsize=16, fontweight="bold")

        # Date tick helpers
        date_indices = np.arange(len(all_dates))
        n_ticks = min(10, len(all_dates))
        tick_step = max(1, len(all_dates) // n_ticks)
        tick_positions = date_indices[::tick_step]
        tick_labels = [all_dates[i] for i in tick_positions]

        # (1,1) Daily annotation count bar chart
        ax = axes[0, 0]
        bars = ax.bar(date_indices, all_counts, color="steelblue", alpha=0.8)
        ax.set_title("Daily Annotations")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        # Add count labels on bars
        if all_counts:
            max_count = max(all_counts)
            threshold = max_count * 0.7
            for bar, count in zip(bars, all_counts):
                if count == 0:
                    continue
                if count > threshold:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - max_count * 0.02,
                            str(count), ha="center", va="top", fontsize=9, color="black")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_count * 0.01,
                            str(count), ha="center", va="bottom", fontsize=9, color="black")

        # (1,2) Cumulative + moving average
        ax = axes[0, 1]
        ax.fill_between(date_indices, cumsum, alpha=0.3, color="steelblue")
        ax.plot(date_indices, cumsum, color="steelblue", linewidth=2, label="Cumulative")
        ax.plot(
            date_indices, moving_avg, color="orangered",
            linewidth=2, linestyle="--", label=f"{window}-day MA",
        )
        ax.set_title("Cumulative & Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)

        # (1,3) Label distribution (horizontal bar)
        ax = axes[0, 2]
        if label_counter:
            top_labels = label_counter.most_common(15)
            labels_list = [item[0] for item in reversed(top_labels)]
            counts_list = [item[1] for item in reversed(top_labels)]
            y_pos = np.arange(len(labels_list))
            ax.barh(y_pos, counts_list, color="steelblue", alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_list, fontsize=8)
            ax.set_xlabel("Count")
        title = "Label Distribution"
        if len(label_counter) > 15:
            title += " (Top 15)"
        ax.set_title(title)

        # (2,1) Shapes per file histogram
        ax = axes[1, 0]
        if shapes_per_file:
            ax.hist(shapes_per_file, bins=min(30, max(5, len(set(shapes_per_file)))),
                    color="steelblue", alpha=0.8, edgecolor="white")
            mean_v = np.mean(shapes_per_file)
            ax.axvline(mean_v, color="orangered", linestyle="--",
                       label=f"Mean: {mean_v:.1f}")
            ax.legend(fontsize=8)
        ax.set_title("Shapes per File")
        ax.set_xlabel("Number of Shapes")
        ax.set_ylabel("Files")

        # (2,2) Vertices per file histogram
        ax = axes[1, 1]
        if vertices_per_file:
            ax.hist(vertices_per_file, bins=min(30, max(5, len(set(vertices_per_file)))),
                    color="steelblue", alpha=0.8, edgecolor="white")
            mean_v = np.mean(vertices_per_file)
            ax.axvline(mean_v, color="orangered", linestyle="--",
                       label=f"Mean: {mean_v:.1f}")
            ax.legend(fontsize=8)
        ax.set_title("Vertices per File")
        ax.set_xlabel("Number of Vertices")
        ax.set_ylabel("Files")

        # (2,3) Area ratio per file histogram
        ax = axes[1, 2]
        if area_per_file:
            ax.hist(area_per_file, bins=min(30, max(5, len(set(area_per_file)))),
                    color="steelblue", alpha=0.8, edgecolor="white")
            mean_v = np.mean(area_per_file)
            ax.axvline(mean_v, color="orangered", linestyle="--",
                       label=f"Mean: {mean_v:.1f}%")
            ax.legend(fontsize=8)
        ax.set_title("BBox Area Ratio per File")
        ax.set_xlabel("Area (%)")
        ax.set_ylabel("Files")

        plt.tight_layout()

        try:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Progress stats saved to: {filename}")
            self.show_status_message(
                self.tr("Progress stats saved to %s") % filename
            )
        except Exception as e:
            plt.close(fig)
            self.errorMessage(
                self.tr("Export Error"),
                self.tr("Failed to save progress stats: %s") % str(e),
            )

    def _open_config_file(self) -> None:
        if self._config_file is None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Config File"),
                self.tr(
                    "Configuration was provided as a YAML expression via "
                    "command line.\n\n"
                    "To use the preferences editor, start Labelme with a config file:\n"
                    "  labelme --config ~/.labelmerc"
                ),
            )
            return
        config_file: Path = self._config_file

        system: str = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", "-t", config_file])
        elif system == "Windows":
            os.startfile(config_file)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", config_file])

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def _can_continue(self) -> bool:
        if not self._is_changed:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            if not self.saveFile():
                # Save failed or the user cancelled the file dialog: stay on
                # this document instead of discarding the changes.
                return False
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def removeSelectedPoint(self):
        shape = self.canvas.prevhShape
        if (
            shape is not None
            and shape.shape_type == "polygon"
            and len(shape.points) <= 3
        ):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Attention"),
                self.tr("ポリゴンの頂点を3点未満にはできません。"),
            )
            return
        # The canvas repaints the vertex's old and new area itself, and emits
        # shapeMoved (-> setDirty) when it actually removed something. It also
        # refuses to empty a shape (canRemovePoint), so there is no
        # "shape ran out of points" case to clean up here.
        self.canvas.removeSelectedPoint()

    def deleteSelectedShape(self):
        if not self.skipDeleteConfirmCheckbox.isChecked():
            yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            msg = self.tr(
                "You are about to permanently delete {} polygons, "
                "proceed anyway?"
            ).format(len(self.canvas.selectedShapes))
            if yes != QtWidgets.QMessageBox.warning(
                self, self.tr("Attention"), msg, yes | no, yes
            ):
                return
        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
        # Disable selection-dependent actions since nothing is selected now
        self.actions.delete.setEnabled(False)
        self.actions.duplicate.setEnabled(False)
        self.actions.copy.setEnabled(False)
        self.actions.edit.setEnabled(False)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.actions.redo.setEnabled(self.canvas.isShapeRedoable)
        self._recompute_reference_medians()
        if self.noShapes():
            for action in self.on_shapes_present_actions:
                action.setEnabled(False)

    # Rectangle corners in drawing order, and which of them each setting drops.
    _RECT_TO_TRIANGLE_CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")

    def convertRectanglesToTriangles(self):
        """Replace rectangles with 3-point polygons (e.g. an arrow bbox -> its
        head triangle), dropping the corner named by
        `rect_to_triangle_drop_corner` in the config.

        Works on the selected rectangles, or on every rectangle when nothing
        is selected. Labels, flags and other attributes are carried over and
        the whole conversion is a single undo step.
        """
        drop = self._config.get("rect_to_triangle_drop_corner", "top_left")
        if drop not in self._RECT_TO_TRIANGLE_CORNERS:
            self.errorMessage(
                self.tr("設定エラー"),
                self.tr(
                    "rect_to_triangle_drop_corner が不正です: {}<br>"
                    "有効な値: {}"
                ).format(drop, ", ".join(self._RECT_TO_TRIANGLE_CORNERS)),
            )
            return

        targets = [
            s
            for s in self.canvas.selectedShapes
            if s.shape_type == "rectangle" and len(s.points) == 2
        ]
        if not targets:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("矩形を3点ポリゴンに変換"),
                self.tr("矩形を選択してください。"),
            )
            return

        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "選択中の矩形 {} 個を、{} を除いた3点ポリゴンに変換します。\n"
            "本当に実施しますか？"
        ).format(len(targets), drop)
        if yes != QtWidgets.QMessageBox.warning(
            self, self.tr("確認"), msg, yes | no, no
        ):
            return

        drop_index = self._RECT_TO_TRIANGLE_CORNERS.index(drop)
        converted = 0
        for shape in targets:
            p0, p1 = shape.points[0], shape.points[1]
            left, right = min(p0.x(), p1.x()), max(p0.x(), p1.x())
            top, bottom = min(p0.y(), p1.y()), max(p0.y(), p1.y())
            corners = [
                QtCore.QPointF(left, top),
                QtCore.QPointF(right, top),
                QtCore.QPointF(right, bottom),
                QtCore.QPointF(left, bottom),
            ]
            del corners[drop_index]
            shape.shape_type = "polygon"
            shape.points = corners
            shape.point_labels = [1] * len(corners)
            shape.close()
            shape.touch()
            converted += 1

        self.canvas.sortShapesByArea()
        self.canvas.storeShapes()
        self._update_shape_items_after_convert(targets)
        self.canvas.update()
        self.setDirty()
        self._recompute_reference_medians()
        self.show_status_message(
            self.tr("%d個の矩形を3点ポリゴンに変換しました") % converted
        )

    def _update_shape_items_after_convert(self, shapes) -> None:
        """Refresh the label-list rows of shapes whose geometry changed."""
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            if item is not None:
                self._update_shape_color(shape)
                item.setText(
                    shape.label
                    if shape.group_id is None
                    else f"{shape.label} ({shape.group_id})"
                )

    def deleteAllShapes(self):
        """Delete all shapes in the currently open image after confirmation."""
        n_shapes = len(self.canvas.shapes)
        if n_shapes == 0:
            return
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "開いている画像のポリゴンを全て削除します（{}件）。\n"
            "本当に実施しますか？"
        ).format(n_shapes)
        if yes != QtWidgets.QMessageBox.warning(
            self, self.tr("確認"), msg, yes | no, no
        ):
            return
        # Assign selection directly: everything is deleted right after, so the
        # per-shape label-list sync in shapeSelectionChanged (O(n^2) with
        # scrollToItem repaints) is wasted work on large images.
        self.canvas.selectedShapes = list(self.canvas.shapes)
        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
        self.actions.delete.setEnabled(False)
        self.actions.duplicate.setEnabled(False)
        self.actions.copy.setEnabled(False)
        self.actions.edit.setEnabled(False)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        self.actions.redo.setEnabled(self.canvas.isShapeRedoable)
        self._recompute_reference_medians()
        if self.noShapes():
            for action in self.on_shapes_present_actions:
                action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def _open_dir_with_dialog(self, _value: bool = False) -> None:
        if not self._can_continue():
            return

        defaultOpenDirPath: str
        if self._prev_opened_dir and osp.exists(self._prev_opened_dir):
            defaultOpenDirPath = self._prev_opened_dir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        if not targetDirPath:
            return
        self._import_images_from_dir(root_dir=targetDirPath)
        self._open_next_image()

    @property
    def imageList(self) -> list[str]:
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            assert item
            # Use stored full path (UserRole) if available, else fall back to text
            path = item.data(Qt.UserRole)
            lst.append(path if path else item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = self._labelFileForImage(file)
            # Display as {dir_name}/{file_name}
            dir_name = osp.basename(osp.dirname(file))
            file_name = osp.basename(file)
            display_name = f"{dir_name}/{file_name}" if dir_name else file_name
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(Qt.UserRole, file)  # Store full path
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            is_annotated = (
                QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file)
            )
            if is_annotated:
                try:
                    self._file_mtimes[file] = osp.getmtime(label_file)
                except OSError:
                    pass
            self._setFileItemAnnotated(item, is_annotated)
            self.fileListWidget.addItem(item)

        self._refresh_file_list_colors()
        self._open_next_image()

    def _import_images_from_dir(
        self, root_dir: str | None, pattern: str | None = None
    ) -> None:

        if not self._can_continue() or not root_dir:
            return

        self._prev_opened_dir = root_dir
        self._initially_annotated_files = set()  # Reset when opening new directory
        self._file_mtimes = {}
        self._file_mtime_range = None
        self.filename = None
        self.fileListWidget.clear()
        self.uniqLabelList.clear()

        filenames = _scan_image_files(root_dir=root_dir)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass

        # Show a progress dialog for large directories (heavy JSON scanning).
        # Threshold is deliberately low: setMinimumDuration below keeps the
        # dialog hidden when the scan finishes quickly anyway, so a directory
        # that is slow for its file count still gets feedback.
        progress: QtWidgets.QProgressDialog | None = None
        total = len(filenames)
        if total >= 20:
            progress = QtWidgets.QProgressDialog(
                self.tr("画像ファイルを読み込み中..."),
                None,  # no cancel button
                0,
                total * 2,  # phase 1: label scan, phase 2: item add
                self,
            )
            progress.setWindowTitle(self.tr("読み込み中"))
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(300)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.setValue(0)

        # Phase 1: Collect labels from JSONs AND populate mtime cache in one pass.
        all_labels: set[str] = set()
        for idx, filename in enumerate(filenames):
            label_file = self._labelFileForImage(filename)
            if osp.exists(label_file):
                try:
                    self._file_mtimes[filename] = osp.getmtime(label_file)
                except OSError:
                    pass
                try:
                    with open(label_file, encoding="utf-8") as f:
                        data = json.load(f)
                    for shape in data.get("shapes", []):
                        if shape.get("label"):
                            all_labels.add(shape["label"])
                except Exception:
                    pass
            if progress is not None and (idx & 0x7) == 0:
                progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
        # Add all labels to uniqLabelList in sorted order for consistent colors.
        # Insert all first, then assign colors — so index-based color lookup
        # sees the final sorted positions.
        for label in sorted(all_labels):
            if self.uniqLabelList.find_label_item(label) is None:
                self.uniqLabelList.add_label_item(
                    label=label,
                    color=(0, 255, 0),
                    sorted_insert=True,
                )
        for row in range(self.uniqLabelList.count()):
            item = self.uniqLabelList.item(row)
            if item:
                label = item.data(Qt.UserRole)
                rgb = self._get_rgb_by_label(label=label)
                item.setData(Qt.UserRole + 1, rgb)
                self.uniqLabelList._update_item_text(item, row)
        # Compute mtime range now so per-item coloring during add uses gradient.
        if self._file_mtimes:
            mts = self._file_mtimes.values()
            self._file_mtime_range = (min(mts), max(mts))
        # Phase 2: Add list widget items.
        self.fileListWidget.setUpdatesEnabled(False)
        try:
            for idx, filename in enumerate(filenames):
                label_file = self._labelFileForImage(filename)
                # Display as {dir_name}/{file_name}
                dir_name = osp.basename(root_dir) if root_dir else ""
                file_name = osp.basename(filename)
                display_name = f"{dir_name}/{file_name}" if dir_name else file_name
                item = QtWidgets.QListWidgetItem(display_name)
                item.setData(Qt.UserRole, filename)  # Store full path
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                is_annotated = (
                    QtCore.QFile.exists(label_file)
                    and LabelFile.is_label_file(label_file)
                )
                if is_annotated:
                    self._initially_annotated_files.add(filename)
                self._setFileItemAnnotated(item, is_annotated)
                self.fileListWidget.addItem(item)
                if progress is not None and (idx & 0x7) == 0:
                    progress.setValue(total + idx)
                    QtWidgets.QApplication.processEvents()
        finally:
            self.fileListWidget.setUpdatesEnabled(True)
        if progress is not None:
            progress.setValue(total * 2)

        # Enable export when files are loaded
        self.actions.exportFileList.setEnabled(self.fileListWidget.count() > 0)
        self.actions.progressStats.setEnabled(self.fileListWidget.count() > 0)
        self._update_nav_button_state()

    def _update_status_stats(self, mouse_pos: QtCore.QPointF) -> None:
        stats: list[str] = []
        stats.append(f"mode={self.canvas.mode.name}")
        stats.append(f"x={mouse_pos.x():6.1f}, y={mouse_pos.y():6.1f}")
        pixel = self.canvas.getPixelInfo(mouse_pos)
        if pixel is not None:
            r, g, b, gray = pixel
            stats.append(f"R={r} G={g} B={b} Gray={gray}")
        self.status_right.setText(" | ".join(stats))


def _scan_image_files(root_dir: str) -> list[str]:
    extensions: list[str] = [
        f".{fmt.data().decode().lower()}"
        for fmt in QtGui.QImageReader.supportedImageFormats()
    ]

    images: list[str] = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.normpath(osp.join(root, file))
                images.append(relativePath)

    logger.debug("found {:d} images in {!r}", len(images), root_dir)

    # Natural sort using standard library only (avoids natsort crash on some environments)
    try:
        locale.setlocale(locale.LC_COLLATE, "")
    except locale.Error:
        pass  # Use default locale if setting fails

    _num_re = re.compile(r"(\d+)")

    def natsort_like_key(p: str):
        name = Path(p).name
        parts = _num_re.split(name)
        return tuple(
            int(x) if x.isdigit() else locale.strxfrm(x.lower()) for x in parts
        )

    return sorted(images, key=natsort_like_key)
