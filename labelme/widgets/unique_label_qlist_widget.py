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

    def mousePressEvent(self, mouseEvent: QtGui.QMouseEvent) -> None:  # type: ignore
        super().mousePressEvent(mouseEvent)
        if not self.indexAt(mouseEvent.pos()).isValid():
            self.clearSelection()

    def find_label_item(self, label: str) -> QtWidgets.QListWidgetItem | None:
        for row in range(self.count()):
            item = self.item(row)
            if item and item.data(Qt.UserRole) == label:
                return item
        return None

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
            self._update_item_text(item, self.count() - 1)

    def _update_item_text(self, item: QtWidgets.QListWidgetItem, index: int) -> None:
        """Update item text with index."""
        label = item.data(Qt.UserRole)
        color = item.data(Qt.UserRole + 1)
        item.setText(
            f"{index}: {html.escape(label)} "
            f"<font color='#{color[0]:02x}{color[1]:02x}{color[2]:02x}'>●</font>"
        )

    def _refresh_indices(self) -> None:
        """Refresh all item indices after insertion."""
        for row in range(self.count()):
            item = self.item(row)
            if item:
                self._update_item_text(item, row)
