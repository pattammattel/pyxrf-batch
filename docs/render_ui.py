"""Render each tab of a Qt .ui layout to PNG images (offscreen).

Usage:
    QT_QPA_PLATFORM=offscreen python render_ui.py <path-to.ui> <output-dir>

Requires only PyQt5 (no beamline dependencies). Loads the .ui form, selects
each tab in turn, and saves it with QWidget.grab().
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets, uic

UI_PATH = sys.argv[1]
OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

app = QtWidgets.QApplication(sys.argv)

win = QtWidgets.QMainWindow()
uic.loadUi(UI_PATH, win)
win.resize(1100, 720)
win.show()
app.processEvents()


def save(widget, name):
    widget.repaint()
    app.processEvents()
    path = os.path.join(OUT_DIR, name)
    widget.grab().save(path)
    print("saved", path)


save(win, "00-main-window.png")

for tab_widget in win.findChildren(QtWidgets.QTabWidget):
    for i in range(tab_widget.count()):
        tab_widget.setCurrentIndex(i)
        app.processEvents()
        title = tab_widget.tabText(i).strip().replace(" ", "_").replace("/", "-")
        save(win, f"{i + 1:02d}-tab-{title}.png")

print("done")
