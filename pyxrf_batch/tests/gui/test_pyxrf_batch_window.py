import pytest
from PyQt6.QtWidgets import QApplication
from pyxrf_batch.gui.windows.pyxrf_batch_window import pyxrf_batchWindow

@pytest.fixture(scope='module')
def app():
    import sys
    app = QApplication(sys.argv)
    yield app
    app.quit()

def test_window_starts(app):
    window = pyxrf_batchWindow()
    assert window is not None
    assert window.isVisible() is False  # should not be visible until shown
