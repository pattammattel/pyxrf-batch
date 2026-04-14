# conda activate analysis-2019-3.0-hxn-clone2

import sys, os, time, subprocess, logging, h5py, traceback, json
import matplotlib.pyplot as plt
import numpy as np
import time
import pyqtgraph as pg

from functools import wraps
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets, uic, QtTest, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from pyxrf.api import *
from epics import caget
from calcs import *

logger = logging.getLogger()
ui_path = os.path.dirname(os.path.abspath(__file__))


def show_error_message_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(None, "Error", error_message)
            pass
    return wrapper


class xrf_3ID(QtWidgets.QMainWindow):
    def __init__(self):
        super(xrf_3ID, self).__init__()
        uic.loadUi(os.path.join(ui_path, "xrf_xanes_TES_gui.ui"), self)

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

        self.pyxrf_subprocess = None

        self.pb_wd.clicked.connect(self.get_wd)
        self.pb_param.clicked.connect(self.get_param)
        self.pb_ref.clicked.connect(self.get_ref_file)
        self.batchJob = {}

        self.pb_start.clicked.connect(self.runSingleXANESJob)
        self.pb_xrf_start.clicked.connect(lambda:self.create_pyxrf_batch_macro())
        self.pb_xanes_calib.clicked.connect(self.getCalibrationData)
        self.pb_plot_calib.clicked.connect(self.plotCalibration)
        self.pb_save_calib.clicked.connect(self.saveCalibration)

        #batchfiles
        self.pb_addTobBatch.clicked.connect(self.addToXANESBatchJob)
        self.pb_runBatch.clicked.connect(self.runBatchFile)
        self.pb_showBatch.clicked.connect(lambda: self.pte_status.append(str(self.batchJob)))
        self.pb_clear_batch.clicked.connect(lambda: self.batchJob.clear())

        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        self.pb_close_plots.clicked.connect(self.close_all_plots)

        self.pb_scan_meta.clicked.connect(self.print_metadata)
        self.pb_scan_dets.clicked.connect(self.print_dets)
        
        
        #load previous config

        line_edits =  [self.le_XRFBatchSID,
                        self.le_wd,
                        self.le_param,
                        self.le_startid,
                        self.le_lastid,
                        self.xanes_elem,
                        self.alignment_elem]


        for le in line_edits:
            le.textEdited.connect(self.save_config)

        self.load_config(os.path.join(ui_path,"config_file.json"))

        
        #threds
        self.scan_thread = QThread()
        self.scan_sts_thread = QThread()
        self.xanes_thread = QThread()
        self.h5thread = QThread()

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum  {self.threadpool.maxThreadCount()} threads")


        self.show()

    def __del__(self):
        import sys
        # Restore sys.stdout
        sys.stdout = sys.__stdout__


    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()


    def errorOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()


    def get_wd(self):
        folder_path = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(folder_path))

    def get_param(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'json file (*.json)')
        self.le_param.setText(str(file_name[0]))

    def get_ref_file(self):
        file_name = QFileDialog().getOpenFileName(self, "Open file", '', 'ref file (*.txt, *.nor, *.csv)')
        self.le_ref.setText(str(file_name[0]))


    def save_config(self):

        self.config = {"xrf_scan_range":self.le_XRFBatchSID.text(),
                       "wd":self.le_wd.text(),
                       "param_file":self.le_param.text(),
                       "xanes_start_id":self.le_startid.text(),
                       "xanes_end_id":self.le_lastid.text(),
                       "xanes_elem":self.xanes_elem.text(),
                       "alignment_elem":self.alignment_elem.text()
                       }

        with open(os.path.join(ui_path, "config_file.json"), "w") as fp:

            json.dump(self.config, fp, indent = 4)

    @show_error_message_box
    def load_config(self, json_file):

        if json_file:

            with open(json_file, 'r') as fp:
                self.config = json.load(fp)
            
                self.le_XRFBatchSID.setText(self.config["xrf_scan_range"]),
                self.le_wd.setText(self.config["wd"]),
                self.le_param.setText(self.config["param_file"]),
                self.le_startid.setText(self.config["xanes_start_id"]),
                self.le_lastid.setText(self.config["xanes_end_id"]),
                self.xanes_elem.setText(self.config["xanes_elem"]),
                self.alignment_elem.setText(self.config["alignment_elem"])


        else:
            pass

    @show_error_message_box
    def parseScanRange(self,str_scan_range):
        scanNumbers = []
        slist = str_scan_range.split(",")
        #print(slist)
        for item in slist:
            if "-" in item:
                slist_s, slist_e = item.split("-")
                print(slist_s, slist_e)
                scanNumbers.extend(list(np.linspace(int(slist_s.strip()), 
											   int(slist_e.strip()), 
											   int(slist_e.strip())-int(slist_s.strip())+1)))
            else:
                scanNumbers.append(int(item.strip()))
        
        return np.int_(scanNumbers)
    
    @show_error_message_box
    def create_pyxrf_batch_macro(self):

        cwd = self.le_wd.text()
        all_sid = self.parseScanRange(self.le_XRFBatchSID.text())
        QtTest.QTest.qWait(500)
        self.pte_status.append(f"scans to process {all_sid}")
        QtTest.QTest.qWait(500)

        h5Param = {'sidList':all_sid,
                   'wd':cwd,
                   'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
                    "xrfParam":self.le_param.text(),
                    "norm" :self.le_sclr_2.text(),
                    "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
                    "XRFfit":self.rb_xrf_fit.isChecked()
                  }


        if self.rb_make_hdf.isChecked():
            self.h5thread = loadh5(h5Param)
            self.h5thread.start()

        elif not self.rb_make_hdf.isChecked() and self.rb_xrf_fit.isChecked():
            
            xrf_batch_param_dict = {"sid_i":all_sid[0],
                                    "sid_f":all_sid[-1],
                                    "wd":cwd,
                                    "xrfParam":self.le_param.text(),
                                     "norm" :self.le_sclr_2.text(),
                                     "saveXRFTiff": self.rb_saveXRFTiff.isChecked()}
            
            self.pyxrfBatchThread = xrfBatchThread(xrf_batch_param_dict)
            self.pyxrfBatchThread.start()

            '''
            for sid in all_sid:
                fname = f"scan2D_{int(sid)}.h5"
                if os.path.isfile(os.path.join(cwd,fname)):
                    self.xrfThread(sid)
                else:
                    print(f"{fname} not found")
            '''
        else:
            pass

    def stopXRFBatch(self):
        self.h5thread.quit()

    def xrfThread(self,sid):

        xrfParam = {
            "sid":sid, 
            "wd":self.le_wd.text(),
            "xrfParam":self.le_param.text(),
            "norm":self.le_sclr_2.text(),
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked()}

        self.xrf_thread = XRFFitThread(xrfParam)
        self.xrf_thread.start()
        
    @show_error_message_box
    def createParamDictXANES(self):

        cwd = self.le_wd.text()
        param = self.le_param.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        ref = self.le_ref.text()
        fit_method = self.cb_fittin_method.currentText()
        elem = self.xanes_elem.text()
        align_elem = self.alignment_elem.text()
        e_shift = float(self.energy_shift.text())
        admm_lambda = int(self.nnls_lamda.text())
        work_flow = self.cb_process.currentText()
        norm = self.le_sclr.text()
        save_all = self.ch_b_save_all_tiffs.isChecked()
        pre_edge = self.ch_b_baseline.isChecked()
        align = self.cb_align.isChecked()

        build_xanes_map_param = {}
        build_xanes_map_param["cwd"] = self.le_wd.text()
        pre_edge = self.ch_b_baseline.isChecked()
        align = self.cb_align.isChecked()

        build_xanes_map_param = {}
        build_xanes_map_param["cwd"] = self.le_wd.text()
        build_xanes_map_param["param"] = self.le_param.text()
        build_xanes_map_param["last_sid"] = int(self.le_lastid.text())
        build_xanes_map_param["first_sid"] = int(self.le_startid.text())
        build_xanes_map_param["ref"] = self.le_ref.text()
        build_xanes_map_param["fit_method"] = self.cb_fittin_method.currentText()
        build_xanes_map_param["elem"] = self.xanes_elem.text()
        build_xanes_map_param["align_elem"] = self.alignment_elem.text()
        build_xanes_map_param["e_shift"] = float(self.energy_shift.text())
        build_xanes_map_param["admm_lambda"] = int(self.nnls_lamda.text())
        build_xanes_map_param["work_flow"] = self.cb_process.currentText()
        build_xanes_map_param["norm"] = self.le_sclr.text()
        build_xanes_map_param["save_all"] = self.ch_b_save_all_tiffs.isChecked()
        build_xanes_map_param["pre_edge"] = self.ch_b_baseline.isChecked()
        build_xanes_map_param["align"] = self.cb_align.isChecked()

        self.pte_status.append(str(build_xanes_map_param))

        return build_xanes_map_param

    def addToXANESBatchJob(self):
        self.batchJob[f"job_{len(self.batchJob)+1}"] = self.createParamDictXANES()
        self.pte_status.append(str(self.batchJob))

    @show_error_message_box
    def runBatchFile(self):
        if self.batchJob:
            for value in self.batchJob.values():
                #plt.close('all')
                self.create_xanes_macro(value)

    def create_xanes_macro(self,param_dict):

        self.xanes_thread = XANESProcessing(param_dict)
        self.xanes_thread.start()
        
        '''
        build_xanes_map(param_dict["first_sid"], param_dict["last_sid"], wd=param_dict["cwd"],
                        xrf_subdir=param_dict["cwd"], xrf_fitting_param_fln=param_dict["param"],
                        scaler_name=param_dict["norm"], sequence=param_dict["work_flow"],
                        ref_file_name=param_dict["ref"], fitting_method=param_dict["fit_method"],
                        emission_line=param_dict["elem"], emission_line_alignment=param_dict["align_elem"],
                        incident_energy_shift_keV=(param_dict["e_shift"] * 0.001),
                        subtract_pre_edge_baseline = param_dict["pre_edge"],
                        alignment_enable = param_dict["align"], output_save_all=param_dict["save_all"],
                        use_incident_energy_from_param_file=True )
        '''
    
    def runSingleXANESJob(self):
        params = self.createParamDictXANES()
        self.create_xanes_macro(params)

    def open_pyxrf(self):
        os.system('gnome-terminal --tab --command pyxrf --active')
        #self.pyxrf_subprocess = subprocess.Popen(['pyxrf'])

    def close_all_plots(self):
        return plt.close('all')



    def closeEvent(self,event):

        for thrd in [self.xanes_thread,self.xrfThread]:
            if not thrd == None:
                if thrd.isRunning():
                    thrd.quit()
                    QtTest.QTest.qWait(500)
                    thrd.quit()
        if not self.pyxrf_subprocess == None:
            if self.pyxrf_subprocess.poll() is None:
                self.pyxrf_subprocess.kill()
        
        sys.exit()

class XANESProcessing(QThread):
    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict

    def run(self):

        build_xanes_map(
            self.paramDict["first_sid"], 
            self.paramDict["last_sid"], 
            wd=self.paramDict["cwd"],
            xrf_subdir=self.paramDict["cwd"], 
            xrf_fitting_param_fln=self.paramDict["param"],
            scaler_name=self.paramDict["norm"], 
            sequence=self.paramDict["work_flow"],
            ref_file_name=self.paramDict["ref"], 
            fitting_method=self.paramDict["fit_method"],
            emission_line=self.paramDict["elem"], 
            emission_line_alignment=self.paramDict["align_elem"],
            incident_energy_shift_keV=(self.paramDict["e_shift"] * 0.001),
            subtract_pre_edge_baseline = self.paramDict["pre_edge"],
            alignment_enable = self.paramDict["align"], 
            output_save_all=self.paramDict["save_all"],
            use_incident_energy_from_param_file=True ,
            skip_scan_types = ['FlyPlan1D']
            )

class loadh5(QThread):
    
    h5loaded = pyqtSignal(int)

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict


    def run(self):
        logger.info("h5 thread started")
        
        for sid in self.paramDict["sidList"]: #filter for 1d
            

            #try to find the data in the db, except pass
            try:
                hdr = db[int(sid)]
                start_doc = hdr["start"]

                #skip 1D scans
                if not start_doc["plan_type"] in ("FlyPlan1D",):
                    
                    
                    #try to make h5, bypass any errors
                    try: 

                        make_hdf(
                            int(sid), 
                            wd = self.paramDict["wd"],
                            file_overwrite_existing = self.paramDict['file_overwrite_existing'],
                            create_each_det = True,
                            skip_scan_types = ['FlyPlan1D']
                            )
                    
                           
                        #self.h5loaded.emit(sid)
                        QtTest.QTest.qWait(50)

                    except:
                        logger.info(f" Failed to load {sid}")
                        pass


                    if self.paramDict["XRFfit"]:

                        fname = f"scan2D_{int(sid)}.h5"

                        if os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5")):

                            fit_pixel_data_and_save(
                                    self.paramDict["wd"],
                                    fname,
                                    param_file_name = self.paramDict["xrfParam"],
                                    scaler_name = self.paramDict["norm"],
                                    save_tiff = self.paramDict["saveXRFTiff"]
                                    )
                    else:
                        pass
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                QMessageBox.critical(None, "Error", error_message)
                logger.error("Trouble finding data from data broker")
                pass

class XRFFitThread(QThread):
    
    def __init__(self, paramDict):
        super().__init__()
        
        self.paramDict = paramDict

    def run(self):
        sid = self.paramDict["sid"]

        fname = f"scan2D_{int(sid)}.h5"

        fit_pixel_data_and_save(
                self.paramDict["wd"],
                fname,
                param_file_name = self.paramDict["xrfParam"],
                scaler_name = self.paramDict["norm"],
                save_tiff = self.paramDict["saveXRFTiff"],
                save_txt = False
                )

        QtTest.QTest.qWait(500)
        self.quit()

        
class xrfBatchThread(QThread):
    def __init__(self, paramDict):
        super().__init__()
            
        self.paramDict = paramDict

    def run(self):
        sid_i = self.paramDict["sid_i"]
        sid_f = self.paramDict["sid_f"]


        pyxrf_batch(self.paramDict["sid_i"], 
            self.paramDict["sid_f"], 
            wd=self.paramDict["wd"], 
            param_file_name=self.paramDict["xrfParam"], 
            scaler_name=self.paramDict["norm"], 
            save_tiff=self.paramDict["saveXRFTiff"],
            save_txt = False,
            ignore_datafile_metadata = True
            )

class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

if __name__ == "__main__":

    formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')
    

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(stream_handler)

    app = QtWidgets.QApplication(sys.argv)
    window = xrf_3ID()
    window.show()
    sys.exit(app.exec_())
