import html

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from .label_list_widget import HTMLDelegate


class _EscapableQListWidget(QtWidgets.QListWidget):
    def keyPressEvent(self, keyEvent: QtGui.QKeyEvent) -> None:  # type: ignore
        super().keyPressEvent(keyEvent)
        if keyEvent.key() == Qt.Key_Escape:
            self.clearSelection()


class UniqueLabelQListWidget(_EscapableQListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setItemDelegate(HTMLDelegate(parent=self))
        self._colormap = None  # Will be set by app.py
        # label -> row, so lookups do not scan the whole list. Rebuilt
        # whenever rows shift (insertion) or the list is cleared.
        self._row_by_label: dict[str, int] = {}

    def setColormap(self, colormap):
        """Set the colormap to use for label colors."""
        self._colormap = colormap

    def mousePressEvent(self, mouseEvent: QtGui.QMouseEvent) -> None:  # type: ignore
        super().mousePressEvent(mouseEvent)
        if not self.indexAt(mouseEvent.pos()).isValid():
            self.clearSelection()

    def find_label_item(self, label: str) -> QtWidgets.QListWidgetItem | None:
        row = self._row_by_label.get(label)
        if row is None or row >= self.count():
            return None
        item = self.item(row)
        if item is not None and item.data(Qt.UserRole) == label:
            return item
        # Index out of sync (rows changed elsewhere): rebuild and retry once.
        self._rebuild_index()
        row = self._row_by_label.get(label)
        return self.item(row) if row is not None else None

    def add_label_item(self, label: str, color: tuple[int, int, int], sorted_insert: bool = False) -> None:
        if self.find_label_item(label):
            raise ValueError(f"Item for label '{label}' already exists")

        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)  # for find_label_item
        # Store color for refresh
        item.setData(Qt.UserRole + 1, color)
        if sorted_insert:
            # Insert in sorted order (by label name)
            insert_row = 0
            for row in range(self.count()):
                existing = self.item(row)
                if existing and existing.data(Qt.UserRole) > label:
                    break
                insert_row = row + 1
            self.insertItem(insert_row, item)
            # Refresh all indices after insertion
            self._refresh_indices()
        else:
            self.addItem(item)
            self._row_by_label[label] = self.count() - 1
            self._update_item_text(item, self.count() - 1)

    def _update_item_text(self, item: QtWidgets.QListWidgetItem, index: int) -> None:
        """Update item text with index. Color is based on row index."""
        label = item.data(Qt.UserRole)
        # Use colormap if available, otherwise fall back to stored color
        if self._colormap is not None and index < len(self._colormap):
            color = tuple(self._colormap[index].tolist())
        else:
            color = item.data(Qt.UserRole + 1)
            if color is None:
                color = (0, 255, 0)  # Default green if color not set
        item.setText(
            f"{index}: {html.escape(label)} "
            f"<font color='#{color[0]:02x}{color[1]:02x}{color[2]:02x}'>●</font>"
        )

    def _refresh_indices(self) -> None:
        """Refresh all item indices after insertion."""
        self._row_by_label = {}
        for row in range(self.count()):
            item = self.item(row)
            if item:
                self._row_by_label[item.data(Qt.UserRole)] = row
                self._update_item_text(item, row)

    def _rebuild_index(self) -> None:
        self._row_by_label = {}
        for row in range(self.count()):
            item = self.item(row)
            if item:
                self._row_by_label[item.data(Qt.UserRole)] = row

    def clear(self) -> None:
        super().clear()
        self._row_by_label = {}

    def get_label_index(self, label: str) -> int | None:
        """Get the row index of a label."""
        row = self._row_by_label.get(label)
        if row is not None and row < self.count():
            item = self.item(row)
            if item is not None and item.data(Qt.UserRole) == label:
                return row
        self._rebuild_index()
        return self._row_by_label.get(label)
