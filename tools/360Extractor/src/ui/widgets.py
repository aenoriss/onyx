from PySide6.QtWidgets import QLabel, QFrame, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QMimeData

class DropZone(QLabel):
    """
    A custom widget that accepts drag-and-drop file inputs.
    """
    files_dropped = Signal(list)
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("dropZone")
        self.setText("Drag & Drop 360° Video Files Here\nor Click to Browse")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setText("Drop files now!")
            self.setProperty("dragActive", True)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setText("Drag & Drop 360° Video Files Here\nor Click to Browse")
        self.setProperty("dragActive", False)
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().polish(self)

        files = []
        for url in event.mimeData().urls():
            files.append(url.toLocalFile())
        
        self.files_dropped.emit(files)
        self.setText(f"{len(files)} file(s) ready")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()