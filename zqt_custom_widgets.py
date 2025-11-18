from PyQt6.QtWidgets import QPushButton, QListWidgetItem
from PyQt6.QtCore import QSize, pyqtSignal

class PreprocessButton(QPushButton):
    start_preprocessing = pyqtSignal()

    def __init__(self, text, audio_preprocessor):
        super().__init__(text=text)
        self.audio_preprocessor = audio_preprocessor
        self.setAcceptDrops(True)
    
    def resizeEvent(self, event):
        size = event.size()
        side = min(size.width(), size.height())

        self.resize(side, side)
        return super().resizeEvent(event)

    def sizeHint(self):
        # size = super().sizeHint()
        # size.setHeight(int((size.width() * 16) / 9))
        # return size
        return QSize(180, 180)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        # Get file name(s) from dropped file(s)
        for url in event.mimeData().urls():
            self.start_preprocessing.emit()
            # TODO: By default url.path() puts \ in front of file path, using slice as temp work around for Windows. Need to fix this.
            self.audio_preprocessor.process_audio(url.path()[1:])
        event.accept()

class SquareButton(QPushButton):
    def __init__(self, text, track_number, audio_manager):
        super().__init__(text=text)
        self.audio_manager = audio_manager
        self.track_number = track_number
        self.setAcceptDrops(True)
    
    def resizeEvent(self, event):
        size = event.size()
        side = min(size.width(), size.height())

        self.resize(side, side)
        return super().resizeEvent(event)

    def sizeHint(self):
        # size = super().sizeHint()
        # size.setHeight(int((size.width() * 16) / 9))
        # return size
        return QSize(120, 120)
    
    def dragEnterEvent(self, event):
        if event:
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        # Get title from event source text and track number from this button
        # Change text of button to song name if loaded successfully
        if event.source() != None:
            if self.audio_manager.load_track(event.source().selectedItems()[0].text(), self.track_number):
                self.setText(event.source().selectedItems()[0].text())
            event.accept()

class SongListItem(QListWidgetItem):
    def __init__(self, text, parent):
        super().__init__(text, parent=parent)


    