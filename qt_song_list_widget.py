from PyQt6.QtWidgets import QListWidgetItem

class SongListItem(QListWidgetItem):
    def __init__(self, text, parent):
        super().__init__(text, parent=parent)


    