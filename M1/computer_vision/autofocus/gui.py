"""Native Qt desktop interface. All image scoring runs outside the UI thread."""
from pathlib import Path
import html
import traceback
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QColor, QPen, QDesktopServices, QFont, QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QComboBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QScrollArea, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QTextBrowser,
    QProgressBar, QGraphicsView, QGraphicsScene, QRubberBand, QAbstractItemView,
    QGridLayout, QFrame, QMessageBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from .dataset import discover_sources, open_dataset, grayscale
from .metrics import METHODS, DEFAULT_METHODS
from .analysis import AnalysisOptions, analyze_dataset, Cancelled, method_name, CONSENSUS, top_indices
from .plotting import curves_figure, fine_figure, statistics_figure, batch_figure, method_figure
from .export import export_results

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = PROJECT / "Documents for autofocusing app"

STYLE = """
QWidget {font-family: 'Apple SD Gothic Neo', 'Segoe UI', sans-serif; font-size: 12px; color: #273a52;}
QMainWindow {background: #edf2f7;}
QFrame#sidebar {background: #f8fafc; border-right: 1px solid #dce4ed;}
QLabel#brand {font-size: 25px; font-weight: 800; color: #15354b;}
QLabel#eyebrow {font-size: 10px; font-weight: 700; color: #07967d;}
QLabel#title {font-size: 23px; font-weight: 700;}
QLabel#muted {color: #718096; font-size: 11px;}
QLabel#warning {color: #8f692b; background: #fff7e8; border: 1px solid #efdfbd; border-radius: 7px; padding: 9px; font-size: 11px;}
QLabel#cardValue {font-size: 21px; font-weight: 700; color: #0e917d;}
QFrame#card {background: white; border: 1px solid #dce4ed; border-radius: 10px;}
QGroupBox {font-weight: 600; border: 1px solid #dce4ed; border-radius: 8px; margin-top: 15px; padding-top: 15px; background: white;}
QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 5px;}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {background: white; border: 1px solid #d6dfeb; border-radius: 6px; padding: 6px; min-height: 18px;}
QLineEdit:focus, QComboBox:focus {border: 1px solid #13a98c;}
QPushButton {background: white; border: 1px solid #d6dfeb; border-radius: 6px; padding: 7px 10px; font-weight: 600;}
QPushButton:hover {background: #e8f6f2; border-color: #94d7c8;}
QPushButton:disabled {color: #a4b1c1; background: #f1f4f8;}
QPushButton#primary {background: #0b9b83; border-color: #0b9b83; color: white; padding: 10px;}
QPushButton#primary:hover {background: #078a73;}
QPushButton#primary:disabled {background: #a5cfc5; border-color: #a5cfc5;}
QPushButton:checked {background: #d6f2e9; border-color: #0b9b83;}
QCheckBox {spacing: 7px; font-weight: 400;}
QListWidget {background: white; border: 1px solid #dce4ed; border-radius: 7px; padding: 3px;}
QListWidget::item {padding: 7px; border-radius: 4px;}
QListWidget::item:selected {background: #ddf3ed; color: #087c68;}
QTabWidget::pane {background: white; border: 1px solid #dce4ed; border-radius: 8px;}
QTabBar::tab {padding: 10px 15px; color: #718096; background: transparent; border-bottom: 3px solid transparent;}
QTabBar::tab:selected {color: #078d75; border-bottom: 3px solid #0b9b83; font-weight: 700;}
QTableWidget {border: 0; background: white; gridline-color: #e6edf4; selection-background-color: #dcf3ec; selection-color: #214639; alternate-background-color: #f7fafc;}
QHeaderView::section {background: #f0f5f9; padding: 8px; border: 0; border-bottom: 1px solid #dce4ed; color: #60738c; font-size: 11px; font-weight: 600;}
QProgressBar {border: 0; border-radius: 4px; background: #e1e8f0; max-height: 9px;}
QProgressBar::chunk {background: #0b9b83; border-radius: 4px;}
QTextBrowser {border: 0; background: white; padding: 10px;}
QScrollArea {border: 0; background: transparent;}
QSplitter::handle {background: #edf2f7;}
QStatusBar {background: #f8fafc; color: #718096; border-top: 1px solid #dce4ed;}
"""


def label(text, name=None, wrap=False):
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


def table(headers):
    item = QTableWidget(0, len(headers))
    item.setHorizontalHeaderLabels(headers)
    item.verticalHeader().hide()
    item.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    item.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    item.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    item.setAlternatingRowColors(True)
    item.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    item.horizontalHeader().setStretchLastSection(True)
    item.setShowGrid(False)
    return item


def fill_table(widget, rows):
    widget.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = QTableWidgetItem(str(value))
            cell.setToolTip(str(value))
            widget.setItem(i, j, cell)


class PlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(2, 2, 2, 2)
        self.canvas = None
        self.reset()

    def reset(self):
        if self.canvas:
            self.canvas.figure.clear()
        while self.box.count():
            child = self.box.takeAt(0).widget()
            if child:
                child.deleteLater()
        self.canvas = None
        self.box.addWidget(label("Run analysis to display the recorded focus curves.", "muted"))

    def set_figure(self, figure):
        if self.canvas:
            self.canvas.figure.clear()
        while self.box.count():
            child = self.box.takeAt(0).widget()
            if child:
                child.deleteLater()
        self.canvas = FigureCanvasQTAgg(figure)
        self.box.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.box.addWidget(self.canvas, 1)
        self.canvas.draw_idle()


class ImageViewer(QGraphicsView):
    roi_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#0c1c2b"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pixmap = None
        self.width_px = self.height_px = 0
        self.roi_rect = None
        self.selecting = False
        self.band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
        self.origin = QPoint()
        self.auto_fit = True

    def display(self, image, roi=(0, 0, 1, 1), edges=False):
        gray = np.clip(grayscale(image), 0, 255).astype(np.uint8)
        if image.ndim == 3:
            if image.dtype != np.uint8:
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if edges:
            rgb = rgb.copy()
            rgb[cv2.Canny(gray, 80, 160) > 0] = (32, 229, 173)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.scene().clear()
        self.pixmap = self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.width_px, self.height_px = w, h
        self.scene().setSceneRect(0, 0, w, h)
        x, y, rw, rh = roi
        if roi != (0, 0, 1, 1):
            self.roi_rect = self.scene().addRect(x*w, y*h, rw*w, rh*h, QPen(QColor("#32e5ad"), 2))
        if self.auto_fit:
            self.fit_image()

    def fit_image(self):
        self.auto_fit = True
        if self.pixmap:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self):
        self.auto_fit = False
        self.resetTransform()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        if not self.pixmap:
            return
        self.auto_fit = False
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        next_scale = self.transform().m11() * factor
        if 0.03 < next_scale < 25:
            self.scale(factor, factor)

    def set_roi_mode(self, enabled):
        self.selecting = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if self.selecting and self.pixmap and event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            self.band.setGeometry(QRect(self.origin, self.origin))
            self.band.show()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selecting and self.band.isVisible():
            self.band.setGeometry(QRect(self.origin, event.position().toPoint()).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.selecting and self.band.isVisible():
            rect = self.mapToScene(self.band.geometry()).boundingRect().intersected(self.sceneRect())
            self.band.hide()
            if rect.width() >= 3 and rect.height() >= 3:
                self.roi_selected.emit((rect.x()/self.width_px, rect.y()/self.height_px, rect.width()/self.width_px, rect.height()/self.height_px))
        else:
            super().mouseReleaseEvent(event)


class AnalysisWorker(QThread):
    progress_changed = Signal(int, str)
    results_ready = Signal(object, object, bool)

    def __init__(self, sources, options, mapping):
        super().__init__()
        self.sources, self.options, self.mapping = sources, options, mapping

    def run(self):
        results, errors, aborted = [], [], False
        for index, source in enumerate(self.sources):
            if self.isInterruptionRequested():
                aborted = True
                break
            try:
                data = open_dataset(source, **self.mapping)
                def report(done, total, name):
                    percent = round(100 * (index + done / total) / len(self.sources))
                    self.progress_changed.emit(percent, f"{data.name} · {done}/{total} · {name}")
                results.append(analyze_dataset(data, self.options, report, self.isInterruptionRequested))
            except Cancelled:
                aborted = True
                break
            except Exception as e:
                errors.append(f"{Path(source).name}: {e}")
                traceback.print_exc()
        self.results_ready.emit(results, errors, aborted)


class ExportWorker(QThread):
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, results, destination):
        super().__init__()
        self.results, self.destination = results, destination

    def run(self):
        try:
            self.ready.emit(str(export_results(self.results, self.destination)))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))



class FormulaPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        box = QVBoxLayout(self)
        box.setContentsMargins(12, 7, 12, 7)
        box.setSpacing(3)
        self.title = label("Mathematical focus measure", "eyebrow")
        self.equation = QLabel()
        self.equation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.equation.setMinimumHeight(48)
        self.description = label("Click a focus measure to inspect its equation.", wrap=True)
        self.parameters = label("", "muted", True)
        self.reference = label("", "muted", True)
        self.reference.setOpenExternalLinks(True)
        for widget in (self.title, self.equation, self.description, self.parameters, self.reference):
            box.addWidget(widget)
        self.setMinimumHeight(160)
        self.setMaximumHeight(225)

    def reset(self):
        self.title.setText("Mathematical focus measure")
        self.equation.clear()
        self.description.setText("Click a focus measure to inspect its equation.")
        self.description.setToolTip("")
        self.parameters.clear()
        self.reference.clear()

    def set_method(self, key):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
        if key == CONSENSUS:
            latex = r"F_C(i)=\frac{1}{K}\sum_{k=1}^{K}\frac{r_k(i)}{n-1}"
            name = "Consensus · mean rank"
            explanation = "Mean normalized rank across selected measures; a comparison aid rather than an independent algorithm."
            parameters = "rk = ascending average rank (0 to n−1); ties share ranks. Correlated methods can dominate agreement."
            reference = "Derived comparison measure"
        else:
            method = next(m for m in METHODS if m.key == key)
            latex, name = method.latex, method.name
            explanation, parameters = method.explanation, method.parameters
            reference = f'<a href="{method.source_url}">{html.escape(method.reference)}</a>' if method.source_url else html.escape(method.reference)
            self.description.setToolTip(method.limitation)
        self.title.setText(name.upper())
        self.description.setText(explanation)
        self.parameters.setText(parameters)
        self.reference.setText(reference)
        figure = Figure(figsize=(7.2, .65), facecolor="white")
        FigureCanvasAgg(figure)
        figure.text(.5, .5, f"${latex}$", ha="center", va="center", fontsize=14 if len(latex)<90 else 11, color="#183d51")
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=130, bbox_inches="tight", pad_inches=.08)
        pixmap = QPixmap()
        pixmap.loadFromData(stream.getvalue())
        self.equation.setPixmap(pixmap.scaled(590, 58, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        figure.clear()


class TopImageCard(QFrame):
    clicked = Signal(int)

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.index = None
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to inspect this ranked image at full size.")
        box = QVBoxLayout(self)
        box.setContentsMargins(7, 6, 7, 6)
        box.setSpacing(3)
        self.heading = label(f"RANK {rank}", "eyebrow")
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(88)
        self.filename = label("—")
        self.filename.setStyleSheet("font-size: 10px; font-weight: 600;")
        self.position = label("", "muted")
        self.score = label("", "muted")
        for widget in (self.heading, self.preview, self.filename, self.position, self.score):
            box.addWidget(widget)

    def set_image(self, result, key, index):
        self.index = index
        if index is None:
            self.preview.clear()
            self.filename.setText("No image")
            self.position.clear()
            self.score.clear()
            return
        entry = result.entries[index]
        gray = np.ascontiguousarray(np.clip(grayscale(result.dataset.load(entry)), 0, 255).astype(np.uint8))
        h, w = gray.shape
        image = QImage(gray.data, w, h, gray.strides[0], QImage.Format.Format_Grayscale8).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(190, 88, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.filename.setText(entry.name)
        self.filename.setToolTip(result.dataset.display_path(entry))
        self.position.setText(f"Image #{entry.sequence+1} / {len(result.dataset.images)} · {entry.distance_mm:.5f} mm")
        self.position.setStyleSheet("font-size: 10px; color:#718096;")
        self.score.setText(f"Focus: {result.scores[key][index]:.7g}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.index is not None:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class FocusWindow(QMainWindow):
    def __init__(self, directory=DEFAULT_DIRECTORY):
        super().__init__()
        self.setWindowTitle("FocusLab · Static Autofocus / CV1A")
        self.setWindowIcon(QIcon(str(PROJECT / "assets/focuslab.svg")))
        self.resize(1540, 1000)
        self.setMinimumSize(1220, 820)
        self.setStyleSheet(STYLE)
        self.results = {}
        self.dataset = self.result = self.current_entry = self.image = None
        self.worker = self.export_worker = None
        self.report_path = None
        self.loading_ticks = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(350)
        self.loading_timer.timeout.connect(self.animate_loading)
        self.build_ui()
        self.path_edit.setText(str(Path(directory).expanduser()))
        self.refresh_sources()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(285)
        outer.addWidget(sidebar)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(15, 17, 15, 12)
        side.addWidget(label("CV1A / CHAPTER 6", "eyebrow"))
        side.addWidget(label("FocusLab", "brand"))
        side.addWidget(label("Recorded SEM stacks · Lab 5.6", "muted"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side.addWidget(scroll, 1)
        controls = QWidget()
        scroll.setWidget(controls)
        box = QVBoxLayout(controls)
        box.setContentsMargins(0, 6, 2, 5)
        self.input_group = QGroupBox("01  Provided image stacks")
        ib = QVBoxLayout(self.input_group)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Full directory path")
        self.path_edit.setToolTip("Paste the original folder or ZIP path and press Enter.")
        self.path_edit.returnPressed.connect(self.refresh_sources)
        ib.addWidget(self.path_edit)
        ib.addLayout(self.buttons([("Folder", self.browse_folder), ("ZIP", self.browse_zip)]))
        self.sources = QListWidget()
        self.sources.setMaximumHeight(137)
        self.sources.currentRowChanged.connect(self.select_source)
        ib.addWidget(self.sources)
        self.source_info = label("", "muted", True)
        ib.addWidget(self.source_info)
        box.addWidget(self.input_group)
        self.method_group = QGroupBox("02  Compare focus measures")
        gb = QVBoxLayout(self.method_group)
        gb.addLayout(self.buttons([("All", lambda: self.select_methods(True)), ("Course + papers", self.default_methods)]))
        self.checks = {}
        for method in METHODS:
            check = QCheckBox(method.name)
            check.setChecked(method.key in DEFAULT_METHODS)
            check.setToolTip(method.explanation + "\n" + method.limitation)
            check.clicked.connect(lambda _checked, key=method.key: self.preview_method(key))
            gb.addWidget(check)
            self.checks[method.key] = check
        box.addWidget(self.method_group)
        self.settings_group = QGroupBox("03  Lab settings")
        form = QGridLayout(self.settings_group)
        self.search_combo = QComboBox()
        for title, key in [("Gradient ascent + verification", "gradient_verified"), ("Multi-start gradient + verification", "multistart"), ("Full recorded-stack maximum", "global")]:
            self.search_combo.addItem(title, key)
        form.addWidget(self.search_combo, 0, 0, 1, 2)
        self.start_fraction = QSpinBox()
        self.start_fraction.setRange(0,100)
        self.start_fraction.setValue(25)
        self.start_fraction.setSuffix(" %")
        self.fit_window = QComboBox()
        for n in (3,5,7,9,11):
            self.fit_window.addItem(f"{n} samples", n)
        self.fit_window.setCurrentIndex(1)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0,5)
        self.sigma.setSingleStep(.2)
        for row,title,widget in [(1,"Initial position",self.start_fraction),(2,"Quadratic window",self.fit_window),(3,"Denoising σ",self.sigma)]:
            form.addWidget(label(title,"muted"),row,0)
            form.addWidget(widget,row,1)
        form.addWidget(label("Default: full original image, σ=0.","muted",True),4,0,1,2)
        box.addWidget(self.settings_group)
        self.wd_group = QGroupBox("04  Working-distance mapping")
        form = QGridLayout(self.wd_group)
        self.metadata = QCheckBox("Use recorded WD metadata")
        self.metadata.setChecked(True)
        self.metadata.toggled.connect(self.mapping_changed)
        form.addWidget(self.metadata,0,0,1,2)
        self.offset = QComboBox()
        self.offset.addItem("First image ↔ WD[0]",0)
        self.offset.addItem("First image ↔ WD[1]",1)
        self.offset.currentIndexChanged.connect(self.mapping_changed)
        form.addWidget(self.offset,1,0,1,2)
        self.distance_start = QDoubleSpinBox()
        self.distance_start.setRange(-1000,1000)
        self.distance_start.setDecimals(4)
        self.distance_start.setValue(10)
        self.distance_start.setSuffix(" mm")
        self.distance_step = QDoubleSpinBox()
        self.distance_step.setDecimals(3)
        self.distance_step.setRange(.001,10000)
        self.distance_step.setValue(10)
        self.distance_step.setSuffix(" µm")
        for row,title,widget in [(2,"Fallback start",self.distance_start),(3,"Fallback step",self.distance_step)]:
            form.addWidget(label(title,"muted"),row,0)
            form.addWidget(widget,row,1)
            widget.valueChanged.connect(self.mapping_changed)
        box.addWidget(self.wd_group)
        box.addStretch()
        self.run_button = QPushButton("Analyze selected stack")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(lambda: self.start_analysis(False))
        side.addWidget(self.run_button)
        self.all_button = QPushButton("Analyze all four stacks")
        self.all_button.clicked.connect(lambda: self.start_analysis(True))
        side.addWidget(self.all_button)
        main = QWidget()
        outer.addWidget(main,1)
        body = QVBoxLayout(main)
        body.setContentsMargins(20,18,20,12)
        body.setSpacing(10)
        top = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("STATIC AUTOFOCUS / LAB 5.6","eyebrow"))
        titles.addWidget(label("Find the sharp image. Compare the methods.","title"))
        top.addLayout(titles)
        top.addStretch()
        self.open_report_button = QPushButton("Open report")
        self.open_report_button.setEnabled(False)
        self.open_report_button.clicked.connect(self.open_report)
        top.addWidget(self.open_report_button)
        self.export_button = QPushButton("Export results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        top.addWidget(self.export_button)
        body.addLayout(top)
        self.busy_panel = QFrame()
        self.busy_panel.setObjectName("card")
        bp = QHBoxLayout(self.busy_panel)
        self.loading_text = label("Analyzing…","eyebrow")
        bp.addWidget(self.loading_text)
        self.progress = QProgressBar()
        self.progress.setRange(0,100)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet("QProgressBar{max-height:22px;min-height:22px;border:1px solid #dce4ed;text-align:center;background:#eef6f3;} QProgressBar::chunk{background:#92daca;}")
        bp.addWidget(self.progress,1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        bp.addWidget(self.cancel_button)
        self.busy_panel.hide()
        body.addWidget(self.busy_panel)
        cards = QHBoxLayout()
        self.dataset_card = self.card(cards,"RECORDED STACK","—")
        self.count_card = self.card(cards,"IMAGES ANALYZED","0")
        self.coarse_card = self.card(cards,"COARSE WD · mm","—")
        self.fine_card = self.card(cards,"FINE ESTIMATE · mm","—")
        body.addLayout(cards)
        self.notice = label("Select a stack and run analysis.","warning",True)
        self.notice.setMaximumHeight(66)
        body.addWidget(self.notice)
        row = QHBoxLayout()
        row.addWidget(label("Selected algorithm"))
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(250)
        self.method_combo.currentIndexChanged.connect(self.change_method)
        row.addWidget(self.method_combo)
        self.focus_value = label("","muted")
        row.addWidget(self.focus_value)
        row.addStretch()
        self.best_button = QPushButton("Show rank 1")
        self.best_button.setEnabled(False)
        self.best_button.clicked.connect(self.show_best)
        row.addWidget(self.best_button)
        body.addLayout(row)
        self.tabs = QTabWidget()
        body.addWidget(self.tabs,1)
        self.build_analysis_tab()
        self.build_comparison_tab()
        batch = QWidget()
        bb = QVBoxLayout(batch)
        self.batch_plot = PlotPanel()
        bb.addWidget(self.batch_plot,1)
        self.batch_table = table(["Stack","Algorithm","Rank-1 file","Image #","Coarse mm","Fine mm","Fit R²"])
        self.batch_table.setMaximumHeight(220)
        bb.addWidget(self.batch_table)
        bb.addWidget(label("Compare scan speeds using the same ROI and preprocessing settings.","muted",True))
        self.tabs.addTab(batch,"Scan-speed comparison")
        files = QWidget()
        fb = QVBoxLayout(files)
        self.files_table = table(["Image #","Filename","WD · mm","Selected score","Original full path"])
        self.files_table.currentCellChanged.connect(self.select_file)
        fb.addWidget(self.files_table)
        fb.addLayout(self.buttons([("Reveal folder",self.reveal_file),("Copy viewed path",self.copy_path)]))
        self.tabs.addTab(files,"Files")
        self.theory = QTextBrowser()
        self.theory.setOpenExternalLinks(True)
        self.theory.setHtml(self.theory_html())
        self.tabs.addTab(self.theory,"Methods & sources")
        self.tabs.currentChanged.connect(self.render_active_tab)
        self.statusBar().showMessage("Ready · Load a provided image stack.")
        action = QAction("Open image folder",self)
        action.setShortcut("Ctrl+O")
        action.triggered.connect(self.browse_folder)
        self.addAction(action)

    def build_analysis_tab(self):
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(8,8,8,8)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lb = QVBoxLayout(left)
        lb.setContentsMargins(0,0,6,0)
        tools = self.buttons([("Fit image",lambda:self.viewer.fit_image()),("100%",lambda:self.viewer.actual_size())])
        self.edges = QCheckBox("Edge overlay")
        self.edges.setToolTip("Canny overlay is visual only; it does not affect scores.")
        self.edges.toggled.connect(self.refresh_image)
        tools.addWidget(self.edges)
        self.roi_button = QPushButton("Draw ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(lambda enabled:self.viewer.set_roi_mode(enabled))
        tools.addWidget(self.roi_button)
        tools.addStretch()
        lb.addLayout(tools)
        self.viewer = ImageViewer()
        self.viewer.roi_selected.connect(self.set_roi)
        lb.addWidget(self.viewer,1)
        self.image_info = label("","muted",True)
        lb.addWidget(self.image_info)
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("Original image path")
        lb.addWidget(self.file_path)
        lb.addWidget(label("TOP 3 · SCORE RANK / ORIGINAL SCAN IMAGE NUMBER","eyebrow"))
        ranks_row = QHBoxLayout()
        self.top_cards = []
        for rank in (1,2,3):
            card = TopImageCard(rank)
            card.clicked.connect(self.show_ranked_image)
            ranks_row.addWidget(card,1)
            self.top_cards.append(card)
        lb.addLayout(ranks_row)
        roi_row = QHBoxLayout()
        roi_row.addWidget(label("ROI %","muted"))
        self.roi_spins = []
        for i,title in enumerate(("X","Y","W","H")):
            spin = QDoubleSpinBox()
            spin.setRange(0,100)
            spin.setDecimals(1)
            spin.setValue(0 if i<2 else 100)
            spin.setPrefix(title+" ")
            spin.valueChanged.connect(self.refresh_image)
            roi_row.addWidget(spin)
            self.roi_spins.append(spin)
        reset = QPushButton("Full")
        reset.clicked.connect(lambda:self.set_roi((0,0,1,1)))
        roi_row.addWidget(reset)
        lb.addLayout(roi_row)
        split.addWidget(left)
        right = QWidget()
        rb = QVBoxLayout(right)
        rb.setContentsMargins(6,0,0,0)
        self.formula_panel = FormulaPanel()
        rb.addWidget(self.formula_panel)
        self.detail_plot = PlotPanel()
        rb.addWidget(self.detail_plot,1)
        self.fit_status = label("","muted",True)
        self.fit_status.setMaximumHeight(58)
        rb.addWidget(self.fit_status)
        split.addWidget(right)
        split.setSizes([660,610])
        box.addWidget(split,1)
        self.tabs.addTab(page,"Selected algorithm")

    def build_comparison_tab(self):
        page = QWidget()
        box = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(label("Display"))
        self.compare_view = QComboBox()
        self.compare_view.addItems(["All focus curves","Distributions & rank correlation"])
        self.compare_view.currentIndexChanged.connect(lambda _:self.render_active_tab(1))
        controls.addWidget(self.compare_view)
        controls.addStretch()
        controls.addWidget(label("Order table by"))
        self.compare_order = QComboBox()
        self.compare_order.addItems(["Peak separation (larger first)","Scoring time (faster first)","Local quadratic R² (larger first)"])
        self.compare_order.currentIndexChanged.connect(self.comparison_table_refresh)
        controls.addWidget(self.compare_order)
        box.addLayout(controls)
        self.comparison_note = label("","muted",True)
        box.addWidget(self.comparison_note)
        self.compare_plot = PlotPanel()
        box.addWidget(self.compare_plot,1)
        self.result_table = table(["Algorithm","Rank-1 image #","Rank-1 file","Coarse mm","Fine mm","Peak gap %","Median ms/image","Fit R²"])
        self.result_table.setMaximumHeight(265)
        self.result_table.cellClicked.connect(self.select_method_row)
        box.addWidget(self.result_table)
        box.addWidget(label("A larger peak gap or fit R² does not establish focus accuracy. Timing excludes file loading and common preprocessing. Click a row to inspect its top 3 images.","muted",True))
        self.tabs.addTab(page,"Compare algorithms")

    @staticmethod
    def buttons(items):
        row = QHBoxLayout()
        row.setSpacing(5)
        for title,callback in items:
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        return row

    @staticmethod
    def card(layout,title,value):
        frame = QFrame()
        frame.setObjectName("card")
        box = QVBoxLayout(frame)
        box.setContentsMargins(12,8,12,8)
        box.addWidget(label(title,"muted"))
        output = label(value,"cardValue")
        output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(output)
        layout.addWidget(frame,1)
        return output

    def options(self):
        return AnalysisOptions(tuple(k for k,c in self.checks.items() if c.isChecked()),self.search_combo.currentData(),self.start_fraction.value()/100,self.fit_window.currentData(),self.sigma.value(),tuple(sp.value()/100 for sp in self.roi_spins))

    def mapping(self):
        return dict(start_mm=self.distance_start.value(),step_um=self.distance_step.value(),use_metadata=self.metadata.isChecked(),metadata_offset=self.offset.currentData())

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self,"Choose the provided stacks folder",self.path_edit.text())
        if path:
            self.path_edit.setText(path)
            self.refresh_sources()

    def browse_zip(self):
        path,_ = QFileDialog.getOpenFileName(self,"Choose a recorded stack ZIP",self.path_edit.text(),"Image stack (*.zip)")
        if path:
            self.path_edit.setText(path)
            self.refresh_sources()

    def refresh_sources(self):
        self.sources.blockSignals(True)
        self.sources.clear()
        found = discover_sources(self.path_edit.text())
        for path in found:
            item = QListWidgetItem(path.name + (" · ZIP" if path.is_file() else ""))
            item.setData(Qt.ItemDataRole.UserRole,str(path))
            item.setToolTip(str(path))
            self.sources.addItem(item)
        self.sources.blockSignals(False)
        self.source_info.setText(f"{len(found)} recorded stacks · original paths")
        self.path_edit.setToolTip(self.path_edit.text())
        self.path_edit.setCursorPosition(0)
        if found:
            self.sources.setCurrentRow(next((i for i,p in enumerate(found) if "spd5" in p.name),0))
        else:
            self.clear_display()
            self.notice.setText("No image stacks found. Check the provided folder path.")

    def mapping_changed(self,*_):
        if hasattr(self,"distance_step") and hasattr(self,"tabs"):
            self.select_source(self.sources.currentRow())

    def clear_display(self):
        self.dataset = self.result = self.current_entry = self.image = None
        self.viewer.scene().clear()
        self.viewer.pixmap = None
        self.file_path.clear()
        self.image_info.clear()
        self.focus_value.clear()
        self.fit_status.clear()
        self.method_combo.clear()
        self.formula_panel.reset()
        first = next((key for key, check in self.checks.items() if check.isChecked()), None)
        if first:
            self.formula_panel.set_method(first)
        self.result_table.setRowCount(0)
        self.files_table.setRowCount(0)
        self.detail_plot.reset()
        self.compare_plot.reset()
        self.notice.setToolTip("")
        self.dataset_card.setText("—")
        self.count_card.setText("0")
        self.coarse_card.setText("—")
        self.fine_card.setText("—")
        self.best_button.setEnabled(False)
        for card in self.top_cards:
            card.set_image(None,None,None)

    def select_source(self,index):
        item = self.sources.item(index)
        if not item:
            return
        self.clear_display()
        try:
            source = item.data(Qt.ItemDataRole.UserRole)
            data = open_dataset(source,**self.mapping())
            self.dataset = data
            self.dataset_card.setText(data.name)
            self.count_card.setText(str(len(data.images)))
            cached = self.results.get(source)
            if cached and [e.distance_mm for e in cached.dataset.images] == [e.distance_mm for e in data.images]:
                self.set_result(cached)
            else:
                self.notice.setText(" · ".join(data.warnings) if data.warnings else "Stack loaded. Select methods and run analysis.")
                self.populate_files()
                self.display_entry(data.images[0])
            self.statusBar().showMessage(f"{data.source} · {data.position_source}")
        except Exception as error:
            self.notice.setText(str(error))

    def select_methods(self,checked):
        for check in self.checks.values():
            check.setChecked(checked)

    def preview_method(self,key):
        if self.result:
            index = self.method_combo.findData(key)
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        else:
            self.formula_panel.set_method(key)
            self.tabs.setCurrentIndex(0)

    def default_methods(self):
        for key,check in self.checks.items():
            check.setChecked(key in DEFAULT_METHODS)

    def start_analysis(self,all_sources=False):
        if self.worker and self.worker.isRunning() or self.export_worker and self.export_worker.isRunning():
            return
        item = self.sources.currentItem()
        sources = [self.sources.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.sources.count())] if all_sources else ([item.data(Qt.ItemDataRole.UserRole)] if item else [])
        if not sources:
            self.notice.setText("Choose a valid recorded image stack first.")
            return
        opts = self.options()
        if not opts.methods:
            self.notice.setText("Select at least one focus measure.")
            return
        x,y,w,h = opts.roi
        if w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001:
            self.notice.setText("ROI must stay inside the image: X+W and Y+H must not exceed 100%.")
            return
        self.progress.setValue(0)
        self.set_busy(True)
        self.statusBar().showMessage(f"Analyzing {len(sources)} stack(s), {len(opts.methods)} algorithms…")
        self.worker = AnalysisWorker(sources,opts,self.mapping())
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.results_ready.connect(self.analysis_done)
        self.worker.finished.connect(lambda:self.set_busy(False))
        self.worker.start()

    def set_busy(self,busy):
        self.run_button.setEnabled(not busy)
        self.all_button.setEnabled(not busy)
        for group in (self.input_group,self.method_group,self.settings_group,self.wd_group):
            group.setEnabled(not busy)
        self.export_button.setEnabled(not busy and bool(self.results))
        self.busy_panel.setVisible(busy)
        if busy:
            self.loading_timer.start()
            self.loading_text.setText("Analyzing…")
        else:
            self.loading_timer.stop()

    def animate_loading(self):
        self.loading_ticks = (self.loading_ticks+1)%4
        self.loading_text.setText("Analyzing" + "."*(self.loading_ticks+1))

    def update_progress(self,value,message):
        self.progress.setValue(value)
        self.progress.setFormat(f"{value}% · {message}")
        self.statusBar().showMessage(message)

    def cancel_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.loading_text.setText("Stopping…")
            self.statusBar().showMessage("Stopping after the current image.")

    def analysis_done(self,results,errors,aborted):
        for result in results:
            self.results[str(result.dataset.source)] = result
        self.select_source(self.sources.currentRow())
        self.batch_table_refresh()
        if errors:
            self.notice.setText(" | ".join(errors))
        elif aborted:
            self.statusBar().showMessage(f"Cancelled. {len(results)} completed stacks retained.")
        else:
            self.progress.setValue(100)
            self.statusBar().showMessage(f"Complete · {len(results)} stacks · {sum(len(r.entries) for r in results)} images · {sum(r.elapsed_seconds for r in results):.2f} s")
        self.render_active_tab(self.tabs.currentIndex())

    def set_result(self,result):
        self.result,self.dataset = result,result.dataset
        self.dataset_card.setText(result.dataset.name)
        self.count_card.setText(f"{len(result.entries)} / {len(result.dataset.images)}")
        self.notice.setText(" · ".join(result.warnings[:2]) if result.warnings else "Analysis complete. Inspect rank 1–3 and compare methods in the next tab.")
        self.notice.setToolTip("\n".join(result.warnings))
        self.method_combo.blockSignals(True)
        for key in result.scores:
            self.method_combo.addItem(method_name(key),key)
        self.method_combo.blockSignals(False)
        self.best_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.change_method()
        self.comparison_table_refresh()

    def change_method(self,*_):
        if not self.result or not self.method_combo.currentData():
            return
        key = self.method_combo.currentData()
        outcome = self.result.methods[key]
        entry = self.result.entries[outcome.search.index]
        self.coarse_card.setText(f"{entry.distance_mm:.6f}")
        fine = outcome.fine.distance_mm
        self.fine_card.setText(f"{fine:.6f}" if fine is not None else "Unavailable")
        self.focus_value.setText(f"Focus {self.result.scores[key][outcome.search.index]:.7g} · image #{entry.sequence+1}")
        self.formula_panel.set_method(key)
        indices = top_indices(self.result.scores[key])
        for i,card in enumerate(self.top_cards):
            card.set_image(self.result,key,indices[i] if i<len(indices) else None)
        r2 = f"R²={outcome.fine.r_squared:.4f}" if outcome.fine.r_squared is not None else "R² unavailable"
        self.fit_status.setText(f"{outcome.fine.status} {r2}. {outcome.search.message}")
        self.fit_status.setToolTip(f"Analysis σ={self.result.options.gaussian_sigma:g}; ROI={self.result.options.roi}; fit={self.result.options.fit_window} samples. Changes apply on the next analysis.")
        self.populate_files()
        self.show_best()
        self.render_active_tab(self.tabs.currentIndex())

    def comparison_table_refresh(self,*_):
        if not self.result:
            return
        pairs = [(k,r) for k,r in self.result.methods.items() if k!=CONSENSUS]
        order = self.compare_order.currentIndex()
        def criterion(pair):
            stats = pair[1].statistics
            if order==0: return -stats["peak_gap_percent"]
            if order==1: return stats.get("median_scoring_ms",float("inf"))
            return -(pair[1].fine.r_squared if pair[1].fine.r_squared is not None else -float("inf"))
        pairs.sort(key=criterion)
        self.comparison_keys = [k for k,_ in pairs]
        rows = []
        for key,out in pairs:
            e = self.result.entries[out.search.index]
            fine,r2 = out.fine.distance_mm,out.fine.r_squared
            timing = out.statistics.get("median_scoring_ms")
            rows.append([method_name(key),e.sequence+1,e.name,f"{e.distance_mm:.6f}",f"{fine:.6f}" if fine is not None else "—",f"{out.statistics['peak_gap_percent']:.4f}",f"{timing:.3f}" if timing is not None else "—",f"{r2:.4f}" if r2 is not None else "—"])
        fill_table(self.result_table,rows)
        if pairs:
            criterion_name = ["largest measured peak separation","fastest scoring","highest local quadratic fit R²"][order]
            self.comparison_note.setText(f"First by {criterion_name}: {method_name(pairs[0][0])}. Compare the actual selected images and curve shapes; these recorded stacks do not supply a ground-truth algorithm winner.")

    def select_method_row(self,row,_):
        if 0<=row<len(self.comparison_keys):
            self.method_combo.setCurrentIndex(self.method_combo.findData(self.comparison_keys[row]))
            self.tabs.setCurrentIndex(0)

    def populate_files(self):
        if not self.dataset:
            return
        key = self.method_combo.currentData()
        values = {e.path:self.result.scores[key][i] for i,e in enumerate(self.result.entries)} if self.result and key else {}
        rows = [[e.sequence+1,e.name,f"{e.distance_mm:.6f}",f"{values[e.path]:.7g}" if e.path in values else "—",self.dataset.display_path(e)] for e in self.dataset.images]
        self.files_table.blockSignals(True)
        fill_table(self.files_table,rows)
        self.files_table.blockSignals(False)

    def select_file(self,row,_column,_oldrow,_oldcolumn):
        if self.dataset and 0<=row<len(self.dataset.images):
            self.display_entry(self.dataset.images[row])
            self.tabs.setCurrentIndex(0)

    def display_entry(self,entry):
        if not self.dataset:
            return
        try:
            self.image = self.dataset.load(entry)
            self.current_entry = entry
            self.viewer.auto_fit = True
            self.refresh_image()
            self.file_path.setText(self.dataset.display_path(entry))
            self.file_path.setToolTip(self.dataset.display_path(entry))
            self.file_path.setCursorPosition(0)
            h,w = self.image.shape[:2]
            self.image_info.setText(f"{entry.name} · image #{entry.sequence+1} / {len(self.dataset.images)} · {w}×{h} px · WD {entry.distance_mm:.6f} mm")
        except Exception as error:
            self.statusBar().showMessage(f"Image display failed: {error}")

    def refresh_image(self,*_):
        if self.image is not None:
            self.viewer.display(self.image,tuple(sp.value()/100 for sp in self.roi_spins),self.edges.isChecked())

    def set_roi(self,roi):
        values = [round(v*100,1) for v in roi]
        values[2],values[3] = min(values[2],100-values[0]),min(values[3],100-values[1])
        for spin,value in zip(self.roi_spins,values):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.refresh_image()
        self.roi_button.setChecked(False)

    def show_best(self):
        if self.result and self.method_combo.currentData():
            self.display_entry(self.result.entries[self.result.methods[self.method_combo.currentData()].search.index])

    def show_ranked_image(self,index):
        if self.result:
            self.display_entry(self.result.entries[index])

    def render_active_tab(self,index):
        if not self.result or not self.method_combo.currentData():
            return
        key = self.method_combo.currentData()
        if index==0:
            self.detail_plot.set_figure(method_figure(self.result,key))
        elif index==1:
            self.compare_plot.set_figure(curves_figure(self.result) if self.compare_view.currentIndex()==0 else statistics_figure(self.result))
            self.comparison_table_refresh()
        elif index==2 and self.results:
            self.batch_plot.set_figure(batch_figure(list(self.results.values())))
            self.batch_table_refresh()

    def batch_table_refresh(self):
        rows = []
        for result in self.results.values():
            for key,out in result.methods.items():
                entry = result.entries[out.search.index]
                rows.append([result.dataset.name,method_name(key),entry.name,entry.sequence+1,f"{entry.distance_mm:.6f}",f"{out.fine.distance_mm:.6f}" if out.fine.distance_mm is not None else "—",f"{out.fine.r_squared:.4f}" if out.fine.r_squared is not None else "—"])
        fill_table(self.batch_table,rows)

    def reveal_file(self):
        if self.dataset:
            folder = self.dataset.source.parent if self.dataset.archived else self.dataset.source
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def copy_path(self):
        QApplication.clipboard().setText(self.file_path.text())
        self.statusBar().showMessage("Original full image path copied.")

    def export(self):
        if not self.results or self.export_worker and self.export_worker.isRunning():
            return
        folder = QFileDialog.getExistingDirectory(self,"Export recorded-stack results",str(PROJECT/"output"))
        if folder:
            self.export_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.all_button.setEnabled(False)
            self.statusBar().showMessage("Writing top-3 images, scores, figures and report…")
            self.export_worker = ExportWorker(list(self.results.values()),folder)
            self.export_worker.ready.connect(self.export_done)
            self.export_worker.failed.connect(lambda error:self.notice.setText(f"Export failed: {error}"))
            self.export_worker.finished.connect(lambda:self.set_busy(False))
            self.export_worker.start()

    def export_done(self,path):
        self.report_path = path
        self.open_report_button.setEnabled(True)
        self.statusBar().showMessage(f"Report saved · {path}")

    def open_report(self):
        if self.report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.report_path))

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                event.ignore()
                return
        if self.export_worker and self.export_worker.isRunning():
            self.statusBar().showMessage("Please close the window after export completes.")
            event.ignore()
            return
        event.accept()

    @staticmethod
    def theory_html():
        methods = "".join(f'<h3>{html.escape(m.name)}</h3><p><code>{html.escape(m.formula)}</code></p><p>{m.explanation} {m.limitation}</p><p>{html.escape(m.parameters)}</p><p style="color:#718096">{f"<a href={m.source_url}>{html.escape(m.reference)}</a>" if m.source_url else html.escape(m.reference)}</p>' for m in METHODS)
        return """<style>body{font-family:sans-serif;color:#273a52;font-size:13px}h1{color:#0b987f}h2,h3{margin-top:22px}code{color:#087d6a}p{line-height:1.6}</style><h1>Static autofocus · CV1A lab 5.6</h1><p>This application analyzes the supplied pollen image stacks. It does not control a microscope.</p><h2>Required coarse-to-fine procedure</h2><ol><li>Calculate a focus measure on every recorded image. Normalized variance is the course baseline.</li><li>Search by centered-difference gradient ascent (Eq. 6.13), halve an unsuccessful step and verify against the full recorded maximum to avoid local peaks.</li><li>Display the coarse sharpest acquired image, its score and working distance. Rank 2 and rank 3 are the next-highest measured scores; the image number is its original 1-based scan sequence.</li><li>Fit a local quadratic q(d)=a₀+a₁d+a₂d² (Eq. 6.10). Estimate its concave vertex using centered, scaled coordinates. Reject boundary, flat, non-concave and out-of-neighbor fits.</li></ol><p>The fine position is interpolated; there is no acquired image at that exact position. Tied scores retain natural scan order and do not establish a uniquely sharpest image.</p><h2>Which algorithm works best on these stacks?</h2><p>Inspect each method's top 3 images, its full focus curve and its local fit. The comparison tab orders methods by peak separation, measured scoring time or fit R². These are separate trade-offs: a larger score is not comparable across formulas; a larger peak gap can reflect noise; a high R² indicates a good quadratic fit, not verified physical focus accuracy. No universal winner is asserted.</p><h2>Recorded distances</h2><p>The supplied sets each contain 199 images and 200 WD values. Metadata begins at 10.26 mm with 5 µm steps, whereas the lab text states 10 mm and 10 µm. Metadata takes priority. The first image is mapped to WD[0] by default; WD[1] can be selected because the source alignment is ambiguous. This changes absolute positions by 5 µm, not image rankings.</p><h2>Research additions</h2><p>Modified Laplacian (LAP2), Gaussian derivative energy (GRA1), and Tenengrad variance (GRA7) follow the operator definitions catalogued in <a href="https://doi.org/10.1016/j.patcog.2012.11.011">Pertuz, Puig &amp; Garcia, Pattern Recognition 46 (2013), 1415–1432</a>. These are explicit discrete, global mean adaptations for the recorded SEM stacks; this app does not reproduce the paper's depth-reconstruction experiments.</p><h2>Implementation conventions</h2><p>All calculations use floating-point grayscale intensities with a fixed 0–255 scale. ROI and optional Gaussian preprocessing are shared by all selected algorithms. Canny overlay is visual only. Course sums are normalized by their valid support sizes. Histogram entropy uses the standard negative sign; the signed Laplacian sum is replaced by variance. Spearman compares image rankings, not accuracy. Timing is median operator time per image on this computer, excluding decoding and common preprocessing. Independent implementations are timed without reusing intermediate results across algorithms.</p><h2>Focus measures</h2>""" + methods + "<h2>Course sources</h2><p>2026–2027_CV1A-lab, §5.6, pp. 29–30.<br>2026–2027_IMG_lectures, Chapter 6, pp. 59–65.</p>"


def run_gui(directory=DEFAULT_DIRECTORY,auto_batch=False):
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("FocusLab")
    app.setOrganizationName("CV1A")
    app.setFont(QFont("Arial",12))
    window = FocusWindow(directory)
    window.show()
    if auto_batch:
        QTimer.singleShot(400,lambda:window.start_analysis(True))
    return app.exec()
