from PyQt6.QtWidgets import QPushButton

class SquareButton(QPushButton):
    def __init__(self, text):
        super().__init__(text=text)
        self.setAcceptDrops(True)
        

    def heightForWidth(self, width):
        return width
    
    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(size.width())
        return size
    
    def dragEnterEvent(self, event):
        if event:
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        self.setText(event.source().selectedItems()[0].text())