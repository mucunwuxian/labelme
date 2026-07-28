from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QStyle


# https://stackoverflow.com/a/2039745/4158863
class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.State_Selected:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.HighlightedText),
            )
        else:
            ctx.palette.setColor(
                QPalette.Text,
                option.palette.color(QPalette.Active, QPalette.Text),
            )

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)

        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)

        thefuckyourshitup_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - thefuckyourshitup_constant
        textRect.setTop(textRect.top() + margin)

        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        thefuckyourshitup_constant = 4
        return QtCore.QSize(
            int(self.doc.idealWidth()),
            int(self.doc.size().height() - thefuckyourshitup_constant),
        )


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super().__init__()
        self.setText(text or "")
        self.setShape(shape)

        self.setCheckable(True)
        self.setCheckState(Qt.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignBottom)

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def setShape(self, shape):
        self.setData(shape, Qt.UserRole)

    def shape(self):
        return self.data(Qt.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.text()}")'


class _ItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.pyqtSignal()

    def removeRows(self, *args, emit_item_dropped=True, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        # Deleting rows on purpose is not a reorder: callers can suppress the
        # notification without blocking the model's own signals, which the
        # shape index depends on.
        if ret and emit_item_dropped:
            self.itemDropped.emit()
        return ret

    def dropMimeData(self, data, action, row: int, column: int, parent):
        # NOTE: By default, PyQt will overwrite items when dropped on them, so we need
        # to adjust the row/parent to insert after the item instead.

        # If row is -1, we're dropping on an item (which would overwrite)
        # Instead, we want to insert after it
        if row == -1 and parent.isValid():
            row = parent.row() + 1
            parent = parent.parent()

        # If still -1, append to end
        if row == -1:
            row = self.rowCount(parent)

        return super().dropMimeData(data, action, row, column, parent)


class LabelListWidget(QtWidgets.QListView):
    itemDoubleClicked = QtCore.pyqtSignal(LabelListWidgetItem)
    itemSelectionChanged = QtCore.pyqtSignal(list, list)

    def __init__(self):
        super().__init__()
        self._selectedItems = []

        self.setWindowFlags(Qt.Window)

        self._model: _ItemModel = _ItemModel()
        self._model.setItemPrototype(LabelListWidgetItem())
        self.setModel(self._model)

        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)

        # shape id -> item, rebuilt lazily. Every model change drops it, so a
        # row added or removed behind this widget's back cannot go unnoticed.
        self._shape_index: dict | None = None
        for signal in (
            self._model.rowsInserted,
            self._model.rowsRemoved,
            self._model.rowsMoved,
            self._model.modelReset,
            self._model.layoutChanged,
        ):
            signal.connect(self._invalidateShapeIndex)
        # Text and colour changes leave the shape mapping intact; only the
        # role that stores the shape matters here.
        self._model.dataChanged.connect(self._onShapeDataChanged)

    def __len__(self):
        return self._model.rowCount()

    def __getitem__(self, i):
        return self._model.item(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def itemDropped(self):
        return self._model.itemDropped

    @property
    def itemChanged(self):
        return self._model.itemChanged

    def itemSelectionChangedEvent(self, selected, deselected):
        selected = [self._model.itemFromIndex(i) for i in selected.indexes()]
        deselected = [self._model.itemFromIndex(i) for i in deselected.indexes()]
        self.itemSelectionChanged.emit(selected, deselected)

    def itemDoubleClickedEvent(self, index):
        self.itemDoubleClicked.emit(self._model.itemFromIndex(index))

    def selectedItems(self):
        return [self._model.itemFromIndex(i) for i in self.selectedIndexes()]

    def scrollToItem(self, item):
        self.scrollTo(self._model.indexFromItem(item))

    def addItem(self, item):
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        self._model.setItem(self._model.rowCount(), 0, item)
        item.setSizeHint(self.itemDelegate().sizeHint(None, None))  # type: ignore[arg-type,union-attr]

    def removeItem(self, item):
        index = self._model.indexFromItem(item)
        self._model.removeRows(index.row(), 1)

    def removeItems(self, items, *, emit_item_dropped=True):
        """Remove several rows, resolving them all before any row moves."""
        rows = set()
        for item in items:
            if item is None:
                continue
            index = self._model.indexFromItem(item)
            if index.isValid():
                rows.add(index.row())
        for row in sorted(rows, reverse=True):
            self._model.removeRows(row, 1, emit_item_dropped=emit_item_dropped)

    def selectItem(self, item):
        index = self._model.indexFromItem(item)
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def selectOnlyItems(self, items):
        """Select exactly these items with a single selection change.

        Same end state as clearSelection() followed by selectItem() for each,
        but it does not emit one selection change per item, which is what made
        selecting thousands of shapes slow.
        """
        selection = QtCore.QItemSelection()
        for item in items:
            index = self._model.indexFromItem(item)
            if index.isValid():
                selection.select(index, index)
        self.selectionModel().select(
            selection, QtCore.QItemSelectionModel.ClearAndSelect
        )

    def _invalidateShapeIndex(self, *args):
        self._shape_index = None

    def _onShapeDataChanged(self, _first, _last, roles=()):
        if not roles or Qt.UserRole in roles:
            self._invalidateShapeIndex()

    def itemsByShape(self):
        """shape id -> item (cached; findItemByShape is a scan per call).

        The returned dict belongs to the widget: treat it as read-only.
        """
        if self._shape_index is None:
            by_shape = {}
            for row in range(self._model.rowCount()):
                item = self._model.item(row, 0)
                if item is not None:
                    by_shape.setdefault(item.shape(), item)
            self._shape_index = by_shape
        return self._shape_index

    def findItemByShape(self, shape):
        return self.itemsByShape().get(shape)

    def clear(self):
        self._model.clear()
