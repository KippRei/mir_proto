from PyQt6.QtWidgets import QPushButton

class SquareButton(QPushButton):
    def __init__(self, text, track_number, audio_manager):
        super().__init__(text=text)
        self.audio_manager = audio_manager
        self.track_number = track_number
        self.setAcceptDrops(True)
    
    # def resizeEvent(self, event):
    #     size = self.sizeHint()
    #     print(f"Width: {size.width()} Height: {size.height()}")

    #     size.setHeight(int((size.width() * 16) / 9))
    #     print(f"Width: {size.width()} Height: {size.height()}")

    #     self.updateGeometry()
    #     return super().resizeEvent(event)

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(int((size.width() * 16) / 9))
        return size
    
    def dragEnterEvent(self, event):
        if event:
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        self.setText(event.source().selectedItems()[0].text())
        self.audio_manager.load_track(self.text(), self.track_number)
        event.accept()