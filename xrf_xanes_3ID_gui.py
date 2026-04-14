# conda activate analysis-2019-3.0-hxn-clone2

import sys, os, time, subprocess, logging, h5py, traceback, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pyqtgraph as pg
import glob
from functools import wraps
from collections import deque
from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
from PyQt5 import QtWidgets, uic, QtTest, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from pyxrf.api import *
from epics import caget
from calcs import *
from pyxrf_tiffs_to_images import *

logger = logging.getLogger()
ui_path = os.path.dirname(os.path.abspath(__file__))


def pyxrf_h5_to_scanlist(folder):
    filelist = glob.glob(os.path.join(folder,"scan2D*.h5"))
    scan_list = []
    for fname in filelist:
        scan_num = os.path.basename(fname).split('_')[1].split('.')[0]
        scan_list.append(int(scan_num))
    return sorted(scan_list)



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

def try_except_pass(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            pass
    return wrapper

@try_except_pass
def run_build_xanes_dict(param_dict):

    save_to_folder = os.path.join(param_dict["cwd"], f'{param_dict["first_sid"]}-{param_dict["last_sid"]}')
    if not os.path.exists(save_to_folder): 
        os.makedirs(save_to_folder)
        
    build_xanes_map(param_dict["first_sid"], 
                    param_dict["last_sid"], 
                    wd=save_to_folder,
                    xrf_subdir=save_to_folder, 
                    xrf_fitting_param_fln=param_dict["param"],
                    scaler_name=param_dict["norm"], 
                    sequence=param_dict["work_flow"],
                    ref_file_name=param_dict["ref"], 
                    fitting_method=param_dict["fit_method"],
                    emission_line=param_dict["elem"], 
                    emission_line_alignment=param_dict["align_elem"],
                    incident_energy_shift_keV=(param_dict["e_shift"] * 0.001),
                    subtract_pre_edge_baseline = param_dict["pre_edge"],
                    alignment_enable = param_dict["align"], 
                    output_save_all=param_dict["save_all"],
                    use_incident_energy_from_param_file=True,
                    skip_scan_types = ['FlyPlan1D','1D_FLY_PANDA' ])

    plt.close()

    if param_dict["align"]:
        build_xanes_map(param_dict["first_sid"], 
                        param_dict["last_sid"], 
                        wd=save_to_folder,
                        xrf_subdir=save_to_folder, 
                        xrf_fitting_param_fln=param_dict["param"],
                        scaler_name=param_dict["norm"], 
                        sequence="build_xanes_map",
                        ref_file_name=param_dict["ref"], 
                        fitting_method=param_dict["fit_method"],
                        emission_line=param_dict["elem"], 
                        emission_line_alignment=param_dict["align_elem"],
                        incident_energy_shift_keV=(param_dict["e_shift"] * 0.001),
                        subtract_pre_edge_baseline = param_dict["pre_edge"],
                        alignment_enable = False, 
                        output_save_all=param_dict["save_all"],
                        use_incident_energy_from_param_file=True, 
                        skip_scan_types = ['FlyPlan1D','1D_FLY_PANDA'])
        
        plt.close()


class xrf_3ID(QtWidgets.QMainWindow):
    def __init__(self):
        super(xrf_3ID, self).__init__()
        uic.loadUi(os.path.join(ui_path, "xrf_xanes_3ID_gui.ui"), self)

        self.pyxrf_subprocess = None
        self.xrf_first_last_thread = QThread()

        self.pb_wd.clicked.connect(self.get_wd)
        self.pb_param.clicked.connect(self.get_param)
        self.pb_ref.clicked.connect(self.get_ref_file)
        self.batchJob = {}

        self.pb_start.clicked.connect(self.runSingleXANESJob)
        self.pb_xrf_start.clicked.connect(lambda:self.create_pyxrf_batch_macro())
        self.pb_stop_batch_xrf.clicked.connect(self.stopXRFBatch)
        #self.pb_live.clicked.connect(self.autoXRFThreadChunkMode)
        self.pb_live.clicked.connect(self.autoXRFThreadFirstLastMode)
        self.pb_stop_live.clicked.connect(self.stopAuto)
        self.pb_load_tracking_file.clicked.connect(self.xrf_fitting_from_trackfile)
        self.pb_start_tracking.clicked.connect(lambda:self.start_to_track_file(self.le_tracking_File.text()))
        self.pb_stop_tracking.clicked.connect(self.stop_file_tracking)
        self.pb_load_multiple_tracking.clicked.connect(self.select_multiple_tracking_files)
        self.pb_start_multiple_tracking.clicked.connect(lambda:self.start_multiple_trackfile_thread(self.le_multiple_tracking.text()))
        self.le_stop_multiple_tracking.clicked.connect(self.stop_multipling_tracking)


        #xanes
        self.pb_xanes_calib.clicked.connect(self.getCalibrationData)
        self.pb_plot_calib.clicked.connect(self.plotCalibration)
        self.pb_save_calib.clicked.connect(self.saveCalibration)
        self.pb_log_file.clicked.connect(self.load_a_logfile)

        #batchfiles
        self.pb_addTobBatch.clicked.connect(self.addToXANESBatchJob)
        self.pb_runBatch.clicked.connect(lambda:self.runBatchFile(
            os.path.join(ui_path,'xanes_batch_params.json'))
            )
        
        self.pb_create_batch_from_log.clicked.connect(
            lambda:self.run_xanes_batch_job_from_logfiles(
            file_filter_key = "nanoXANES",
            file_extention = "csv")
            )
        self.pb_showBatch.clicked.connect(lambda: self.pte_status.append(str(self.batchJob)))
        self.pb_clear_batch.clicked.connect(lambda: self.batchJob.clear())
        self.pb_stop_xanes_batch.clicked.connect(lambda:self.batch_xanes_thread.terminate())

        self.pb_open_pyxrf.clicked.connect(self.open_pyxrf)
        self.pb_close_plots.clicked.connect(self.close_all_plots)

        self.pb_scan_meta.clicked.connect(self.print_metadata)
        self.pb_scan_dets.clicked.connect(self.print_dets)

        self.pb_select_calib_file.clicked.connect(self.get_calib_file)
        
        
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


        #liveupdates
        self.startScanStatusThread()
        self.last_sid = 100

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

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

    def get_calib_file(self):
        calib_fname = QFileDialog().getOpenFileName(self, "Open file", '', 'json file (*.json)')
        if calib_fname[0]:
            self.le_quant_calib_file.setText(calib_fname[0])

    def get_wd(self):
        dirname = QFileDialog().getExistingDirectory(self, "Select Folder")
        self.le_wd.setText(str(dirname))

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
            
            try:
                self.le_XRFBatchSID.setText(self.config["xrf_scan_range"]),
                self.le_wd.setText(self.config["wd"]),
                self.le_param.setText(self.config["param_file"]),
                self.le_startid.setText(self.config["xanes_start_id"]),
                self.le_lastid.setText(self.config["xanes_end_id"]),
                self.xanes_elem.setText(self.config["xanes_elem"]),
                self.alignment_elem.setText(self.config["alignment_elem"])

            except:
                pass


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
        
        return np.int_(sorted(scanNumbers))
    
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
                    "XRFfit":self.rb_xrf_fit.isChecked(),
                    "quant_calib_file":self.le_quant_calib_file.text(),
                    "quant_calib_elem":self.le_qunat_ref_elem.text(),
                    "interpolate_to_uniform_grid":self.rb_inter_uni_grid.isChecked()
                  }


        if self.rb_make_hdf.isChecked():
            self.h5thread = Loadh5AndFit(h5Param)
            self.h5thread.start()

        elif not self.rb_make_hdf.isChecked() and self.rb_xrf_fit.isChecked():
            
            xrf_batch_param_dict = {"sid_i":all_sid[0],
                                    "sid_f":all_sid[-1],
                                    "wd":cwd,
                                    "xrfParam":self.le_param.text(),
                                     "norm" :self.le_sclr_2.text(),
                                     "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
                                     "quant_calib_file":self.le_quant_calib_file.text(),
                                     "quant_calib_elem":self.le_qunat_ref_elem.text(),
                                     "interpolate_to_uniform_grid":self.rb_inter_uni_grid.isChecked()}
            
            self.pyxrfBatchThread = xrfBatchThread(xrf_batch_param_dict)
            self.pyxrfBatchThread.start()

        else:
            pass

    def stopXRFBatch(self):
        try:
            print("Trying to kill the process")
            self.h5thread.requestInterruption()
            self.h5thread.terminate()

            if self.pyxrfBatchThread:

                self.pyxrfBatchThread.requestInterruption()
                self.pyxrfBatchThread.terminate()
            #print("Batch Process Killed")
            
        except: pass
        print("Batch Process Killed")

        
    @show_error_message_box
    def createParamDictXANES(self):

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
        
        norm = self.le_sclr.text()
        if norm == 'None':
            build_xanes_map_param["norm"] = None
        else:
            build_xanes_map_param["norm"] = norm

        build_xanes_map_param["save_all"] = self.ch_b_save_all_tiffs.isChecked()
        build_xanes_map_param["pre_edge"] = self.ch_b_baseline.isChecked()
        build_xanes_map_param["align"] = self.cb_align.isChecked()

        self.pte_status.append(str(build_xanes_map_param))

        return build_xanes_map_param

    def addToXANESBatchJob(self):
        self.batchJob[f"job_{len(self.batchJob)+1}"] = self.createParamDictXANES()
        out_file_ = os.path.join(ui_path,'xanes_batch_params.json')
        with open(out_file_, 'w') as outfile:
            json.dump(self.batchJob, outfile, indent=6)

        outfile.close()
        #self.pte_status.append(str(self.batchJob))

    def show_in_pte(self,str_to_show):
        self.pte_status.append(str(str_to_show))

    def runBatchFile(self, param_file):
        with open(param_file,'r') as infile:
            batch_job = json.load(infile)
        infile.close()

        if batch_job:
            self.xanes_batch_progress.setRange(0,0)

            self.batch_xanes_thread = XANESBatchProcessing(batch_job)
            self.batch_xanes_thread.current_process.connect(self.show_in_pte)
            self.batch_xanes_thread.finished.connect(lambda:self.xanes_batch_progress.setRange(0,100))
            self.batch_xanes_thread.finished.connect(lambda:self.xanes_batch_progress.setValue(100))
            self.batch_xanes_thread.start()


    def export_xanes_batch_param_file(self):
        out_file_path = os.path.join(self.le_wd.text(),"batch_xanes_params.json")
        export_file_name = QFileDialog.getSaveFileName(self,
                                "export_xanes_params",
                                out_file_path,
                                "All Files (*)")
        if export_file_name[0]:
            with open(out_file_path, 'w') as outfile:
                json.dump(self.batchJob, outfile, indent=6)
        else:
            return
        
    def load_a_logfile(self):
        fname,_ = QFileDialog.getOpenFileName(self,"import log file", 
                                             self.le_wd.text(),
                                             "csv file (*.csv)")
        if fname:
            self.le_log_file.setText(fname)
            log = pd.read_csv(fname)
            if "scan_id" in log.columns:
                sid_list = log['scan_id'].dropna().to_numpy(dtype='int')
            else:
                # Find columns starting with "scan" (case-insensitive)
                matching_columns = [col for col in log.columns if col.lower().startswith("scan")]
                if matching_columns:
                    print(f"No 'scan_id' found. Matching columns: {matching_columns}")
                    sid_list = log[str(matching_columns[0])].dropna().to_numpy(dtype='int')
                else:
                    print("No matching columns found.")
            self.le_startid.setText(str(sid_list[0]))
            self.le_lastid.setText(str(sid_list[-1]))
            new_batch_job = {}
            temp_batch_file = self.createParamDictXANES()
            new_batch_job[f"job_{len(new_batch_job)+1}"] = temp_batch_file

            choice = QMessageBox.question(None,'Scans to process',
                                f"The batch job is {new_batch_job}. \n Proceed?", 
                                QMessageBox.Yes |
                                QMessageBox.No, QMessageBox.No)

            if choice == QMessageBox.Yes:
                if new_batch_job:
                    self.xanes_batch_progress.setRange(0,0)
                    self.batch_xanes_thread = XANESBatchProcessing(new_batch_job)
                    self.batch_xanes_thread.current_process.connect(self.show_in_pte)
                    self.batch_xanes_thread.finished.connect(
                        lambda:self.xanes_batch_progress.setRange(0,100)
                        )
                    self.batch_xanes_thread.finished.connect(
                        lambda:self.xanes_batch_progress.setValue(100)
                        )
                    self.batch_xanes_thread.start()
            else:
                return

        
    def create_singlejob_from_logfile(self,fname):
        if fname:
            new_batch_job = {}
            df = pd.read_csv(fname)
            sid_list = df["scan_id"].dropna().to_numpy(dtype = int)

            temp_batch_file = self.createParamDictXANES()
            temp_batch_file['last_sid'] = int(sid_list[-1])
            temp_batch_file['first_sid'] = int(sid_list[0])
            new_batch_job[f"job_{len(new_batch_job)+1}"] = temp_batch_file

            choice = QMessageBox.question(None,'Scans to process',
                                f"The batch job is {new_batch_job}. \n Proceed?", 
                                QMessageBox.Yes |
                                QMessageBox.No, QMessageBox.No)

            if choice == QMessageBox.Yes:
                if new_batch_job:
                    self.xanes_batch_progress.setRange(0,0)
                    self.batch_xanes_thread = XANESBatchProcessing(new_batch_job)
                    self.batch_xanes_thread.current_process.connect(self.show_in_pte)
                    self.batch_xanes_thread.finished.connect(
                        lambda:self.xanes_batch_progress.setRange(0,100)
                        )
                    self.batch_xanes_thread.finished.connect(
                        lambda:self.xanes_batch_progress.setValue(100)
                        )
                    self.batch_xanes_thread.start()
            else:
                return

        
        
    def run_xanes_batch_job_from_logfiles(self, file_filter_key = " ",
                                      file_extention = "csv"):
        

        dirname = QFileDialog.getExistingDirectory(self, 
                                                   "Select Folder", 
                                                   self.le_wd.text(), )
        print(dirname)

        if dirname:
            logfiles = glob.glob(os.path.join(dirname,
                                         f"*.{file_extention}"))

        else:
            return 

        choice = QMessageBox.question(None,'Files Found',
                                      f"Files found are {logfiles}. \n Proceed?", 
                                      QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        
        if choice == QMessageBox.Yes:
            new_batch_job = {}
            for fname in sorted(logfiles):
                log = pd.read_csv(fname)
                if "scan_id" in log.columns:
                    sid_list = log['scan_id'].dropna().to_numpy(dtype='int')
                else:
                    # Find columns starting with "scan" (case-insensitive)
                    matching_columns = [col for col in log.columns if col.lower().startswith("scan")]
                    if matching_columns:
                        print(f"No 'scan_id' found. Matching columns: {matching_columns}")
                        sid_list = log[str(matching_columns[0])].dropna().to_numpy(dtype='int')
                    else:
                        print("No matching columns found.")

                temp_batch_file = self.createParamDictXANES()
                temp_batch_file['last_sid'] = int(sid_list[-1])
                temp_batch_file['first_sid'] = int(sid_list[0])
                new_batch_job[f"job_{len(new_batch_job)+1}"] = temp_batch_file

        else:
            return

        choice = QMessageBox.question(None,'Scans to process',
                                      f"The batch job is {new_batch_job}. \n Proceed?", 
                                      QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)
        
        if choice == QMessageBox.Yes:
            if new_batch_job:
                self.xanes_batch_progress.setRange(0,0)
                self.batch_xanes_thread = XANESBatchProcessing(new_batch_job)
                self.batch_xanes_thread.current_process.connect(self.show_in_pte)
                self.batch_xanes_thread.finished.connect(
                    lambda:self.xanes_batch_progress.setRange(0,100)
                    )
                self.batch_xanes_thread.finished.connect(
                    lambda:self.xanes_batch_progress.setValue(100)
                    )
                self.batch_xanes_thread.start()
        else:
            return

    def runSingleXANESJob(self):
        params = self.createParamDictXANES()
        self.xanes_thread = XANESProcessing(params)
        self.xanes_thread.start()

    def getCalibrationData(self):

        cwd = self.le_wd.text()
        last_sid = int(self.le_lastid.text())
        first_sid = int(self.le_startid.text())
        if self.rb_Loadh5AndFit.isChecked():
            make_hdf(first_sid, last_sid, wd=cwd, 
                     file_overwrite_existing=self.rb_h5Overwrite.isChecked())
        else:
            QtTest.QTest.qWait(1)
            self.pte_status.append("Loading h5 From DataBroker is skipped ")
        #worker2 = Worker(getCalibSpectrum, path_ = cwd)
        #worker2.signals.result.connect(self.print_output)
        #list(map(worker2.signals.finished.connect, [self.thread_complete, self.plotCalibration]))
        self.calib_spec = getCalibSpectrum(path_= cwd)
        QtTest.QTest.qWait(1)
        self.pte_status.append(str("calibration spec available"))
        #np.savetxt(os.path.join(self.le_wd.text(), "calibration_spec.txt"), self.calib_spec)
        self.plotCalibration()

    def plotCalibration(self):
        if self.rb_calib_derivative.isChecked():
            pg.plot(self.calib_spec[:, 0], np.gradient(self.calib_spec[:, 1]),
                    pen = pg.mkPen(pg.mkColor(0,0,255,255), width=3),
                    symbol='o',symbolSize = 6,symbolBrush = 'r', title = "Calibration Spectrum")
        else:
            pg.plot(self.calib_spec[:, 0], self.calib_spec[:, 1],
                    pen = pg.mkPen(pg.mkColor(0,0,255,255), width=3),
                    symbol='o',symbolSize = 6,symbolBrush = 'r', title = "Calibration Spectrum")

    def saveCalibration(self):
        file_name = QFileDialog().getSaveFileName(self, "Save Calibration", '', 'txt file (*.txt)')
        if file_name[0]:
            np.savetxt(file_name[0], self.calib_spec, fmt = '%.5f')
        else:
            pass

    def stopAuto(self):
        self.scan_thread.requestInterruption()
        #self.scan_thread.wait()
        QtTest.QTest.qWait(int(1000))
        self.scan_thread.terminate()
        self.pte_status.clear()
        self.lbl_live_sts_msg.setText("  Live processing is OFF  ")
        self.lbl_live_sts_msg.setStyleSheet("background-color: yellow")
        self.pb_live.setEnabled(True)
        print(f"Thread Running: {self.scan_thread.isRunning()}")

    def liveButtonSts(self, sts):
        self.pb_live.setEnabled(sts)
        self.lbl_live_sts_msg.setText("  Live processing is ON  ")
        self.lbl_live_sts_msg.setStyleSheet("background-color: lightgreen")

    def scanStatusUpdate(self,sts):
        
        if sts == 1:
            self.lbl_scan_status.setText("    Scan in Progress    ")
            self.lbl_scan_status.setStyleSheet("background-color: lightgreen")

        else:
            self.lbl_scan_status.setText("    Run Engine is Idle    ")
            self.lbl_scan_status.setStyleSheet("background-color: yellow")
         
    def startScanStatusThread(self):

        self.scan_sts_thread = scanStatus()
        self.scan_sts_thread.scan_sts.connect(self.scanStatusUpdate)
        self.scan_sts_thread.scan_num.connect(self.sb_scan_number.setValue)
        self.scan_sts_thread.start()

    #not using
    def autoXRFThreadChunkMode(self):

        self.scan_thread = ScanListStream(self.sb_chunk_size.value())
        self.scan_thread.scan_list.connect(self.pyxrf_live_batch_mode) #thread this
        self.scan_thread.enableLiveButton.connect(self.liveButtonSts)
        self.scan_thread.start()
        print(f"Auto XRF Thread Running: {self.scan_thread.isRunning()}")
    
    #using
    def autoXRFThreadFirstLastMode(self):

        self.scan_thread = ScanListStream2(self.xrf_first_last_thread)
        self.scan_thread.first_last_sid_sig.connect(self.pyxrf_live_first_last_mode) #thread this
        self.scan_thread.enableLiveButton.connect(self.liveButtonSts)
        self.scan_thread.start()
        print(f"Auto XRF Thread Running: {self.scan_thread.isRunning()}")

    def handle_returned_1d_list(self, value):
        self.skipped_1d = value
        

    def pyxrf_live_first_last_mode(self, sid_first_last):
        print(f"live process started : {sid_first_last}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)

        sid_list = np.arange(sid_first_last[0],sid_first_last[-1]+1)
        
        cwd = self.le_wd.text()
        param = self.le_param.text()
        # cwd = "/data/users/2024Q3/Commissioning _2024Q3/xrf_live_test"
        # param = "/data/users/current_user/xrf/pyxrf_model_parameters_301387.json"
        
        norm = self.le_sclr_2.text()

        h5Param = {'sidList':sid_list,
            'wd':cwd,
            'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
            "xrfParam":param,
            "norm" :norm,
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "XRFfit":self.rb_xrf_fit.isChecked(),
            "interpolate_grid":self.rb_inter_uni_grid.isChecked()
            }
            
        #self.xrf_batch_thread = Loadh5AndFit(h5Param)

        try: 
            self.skipped_1d
        except AttributeError:
            self.skipped_1d = deque(maxlen=30)

        #print(f"{self.skipped_1d = }")

        self.xrf_first_last_thread = Loadh5AndFitFromListLive(h5Param,self.skipped_1d)
        self.xrf_first_last_thread.start()
        self.xrf_first_last_thread.skipped_1d_scans_sig.connect(self.handle_returned_1d_list)
        #self.xrf_first_last_thread.last_processed.connect(self.sb_last_sid_processed.setValue)



    def pyxrf_live_batch_mode(self, sid_list):
        print(f"live process started : {sid_list}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()

        h5Param = {'sidList':sid_list,
            'wd':cwd,
            'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
            "xrfParam":self.le_param.text(),
            "norm" :norm,
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "XRFfit":self.rb_xrf_fit.isChecked(),
            "interpolate_to_uniform_grid":self.rb_inter_uni_grid.isChecked()
            }
            
        #self.xrf_batch_thread = Loadh5AndFit(h5Param)
        self.xrf_batch_thread = Loadh5AndFitForLive(h5Param)
        self.xrf_batch_thread.start()
        self.xrf_batch_thread.last_processed.connect(self.sb_last_sid_processed.setValue)


    def xrf_fitting_from_trackfile(self):
        tracking_file,_ = QFileDialog().getOpenFileName(self, "Open file",
                                                         '/data/users/current_user', 
                                                         'tracking file(*)')
        if tracking_file:
            self.le_tracking_File.setText(tracking_file)

            choice = QMessageBox.question(None,'Tracking XRF Fitting',
                                f"Tracking File {tracking_file}. \n Start Monitoring and Fitting?", 
                                QMessageBox.Yes |
                                QMessageBox.No, QMessageBox.No)

            if choice == QMessageBox.Yes:
                self.start_to_track_file(tracking_file)

        else: return

    def start_to_track_file(self,tracking_file):
        
        self.trackfile_thread = TrackingFileToScanNumerThreadLive(tracking_file)
        self.trackfile_thread.scan_list_sig.connect(self.pyxrf_track_file_mode) #thread this
        self.trackfile_thread.start()
        print(f"Auto XRF Trackfile Running: {self.trackfile_thread.isRunning()}")

    def stop_file_tracking(self):
        self.trackfile_thread.requestInterruption()
        self.trackfile_thread.terminate()
        self.xrf_batch_tracking_thread.requestInterruption()
        self.xrf_batch_tracking_thread.terminate()

    def pyxrf_track_file_mode(self, sid_list):
        print(f"live process started : {sid_list}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()

        h5Param = {'sidList':sid_list,
            'wd':cwd,
            'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
            "xrfParam":self.le_param.text(),
            "norm" :norm,
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "XRFfit":self.rb_xrf_fit.isChecked(),
            "interpolate_to_uniform_grid":self.rb_inter_uni_grid.isChecked()
            }
            
        #self.xrf_batch_thread = Loadh5AndFit(h5Param)
        self.xrf_batch_tracking_thread = Loadh5AndFitFromList(h5Param)
        self.xrf_batch_tracking_thread.start()

    def select_multiple_tracking_files(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Multiple Files")
        # Set the file mode to allow multiple selections
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()

        self.le_multiple_tracking.setText(str(file_paths))

    def start_multiple_trackfile_thread(self,file_list):

        for tracking_file in eval(file_list):
            if tracking_file.endswith(".csv"):
                log = pd.read_csv(tracking_file).dropna()
                if "scan_id" in log.columns:
                    self.scan_list = log['scan_id'].dropna().to_numpy(dtype='int')
                else:
                    # Find columns starting with "scan" (case-insensitive)
                    matching_columns = [col for col in log.columns if col.lower().startswith("scan")]
                    if matching_columns:
                        print(f"No 'scan_id' found. Matching columns: {matching_columns}")
                        self.scan_list = log[str(matching_columns[0])].dropna().to_numpy(dtype='int')
                    else:
                        print("No matching columns found.")

            elif tracking_file.endswith(".txt"):
                log = np.loadtxt(tracking_file, dtype = int)
                self.scan_list = list(log[:,0])

            self.pte_status.append(f"scans to process {self.scan_list}")

            self.pyxrf_track_file_mode(self.scan_list)

            QMessageBox.info(None, "Done!", "XRF Fitting completed")
            

    def stop_multipling_tracking(self):

        if self.trackfile_thread.isRunning():
            self.trackfile_thread.requestInterruption()
            self.trackfile_thread.terminate()

    #not using
    def pyxrf_live_collector_mode(self, sid_list):
        print(f"live process started : {sid_list}")
        self.pb_live.setToolTip("Disable when live. Clip stop to restart")
        self.pb_live.setEnabled(False)
        
        cwd = self.le_wd.text()
        norm = self.le_sclr_2.text()

        h5Param = {'sidList':sid_list,
            'wd':cwd,
            'file_overwrite_existing':self.rb_h5OverWrite.isChecked(),
            "xrfParam":self.le_param.text(),
            "norm" :norm,
            "saveXRFTiff": self.rb_saveXRFTiff.isChecked(),
            "XRFfit":self.rb_xrf_fit.isChecked(),
            "interpolate_to_uniform_grid":self.rb_inter_uni_grid.isChecked()
            }
            
        #self.xrf_batch_thread = Loadh5AndFit(h5Param)
        self.xrf_batch_tracking_thread = Loadh5AndFitFromList(h5Param)
        self.xrf_batch_tracking_thread.start()
        

    def open_pyxrf(self):
        os.system('gnome-terminal --tab --command pyxrf --active')
        #self.pyxrf_subprocess = subprocess.Popen(['pyxrf'])

    def close_all_plots(self):
        return plt.close('all')

    def print_metadata(self):
        sid = int(self.le_sid_meta.text())
        h = db[sid]
        self.pte_status.clear()
        self.pte_status.append(str(h.start))

    def print_dets(self):
        sid = int(self.le_sid_meta.text())
        h = db[sid]
        self.pte_status.clear()
        self.pte_status.append(str(h.start['detectors']))

    # Thread Signals

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def threadMaker(self, funct):
        # Pass the function to execute
        worker = Worker(funct)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # Execute
        self.threadpool.start(worker)

    def closeEvent(self,event):

        for thrd in [self.scan_thread,self.scan_sts_thread,self.xanes_thread,self.h5thread]:
            if not thrd == None:
                if thrd.isRunning():
                    thrd.requestInterruption()
                    #thrd.wait()
                    thrd.terminate()
                    QtTest.QTest.qWait(100)
                    #thrd.wait()
        if not self.pyxrf_subprocess == None:
            if self.pyxrf_subprocess.poll() is None:
                self.pyxrf_subprocess.kill()
        
        sys.exit()


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error:`tuple` (exctype, value, traceback.format_exc() )
    - result: `object` data returned from processing, anything
    - progress: `tuple` indicating progress metadata
    '''
    start = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        '''
        Initialize the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        self.signals.start.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class ScanListStream(QThread):
    scan_list = pyqtSignal(list)
    enableLiveButton = pyqtSignal(bool)
    def __init__(self, chunk_length):
        super().__init__()
        self.chunk_length = chunk_length
        self.scans_to_process = []

    def run(self):
        self.enableLiveButton.emit(False)
        timeout = time.time() + 60*60   # 60 minute intervals
        timeout_for_live = time.time() + 60*60*24*5 # max 5 days active loop 
        previous_list = [100,200]
        while True:
            QtTest.QTest.qWait(300)
            sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))

            if caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and not sid in self.scans_to_process and sid not in previous_list:
                self.scans_to_process.append(sid)
                logger.info(f"new scan added: {sid}; scan list to process{self.scans_to_process}")
                print(f"new scan added: {sid}; scan list to process = {self.scans_to_process}")
            if len(self.scans_to_process) == self.chunk_length or (time.time() > timeout and self.scans_to_process):
                self.scan_list.emit(self.scans_to_process) #to give the last scan to save to the db-had problems in missing because of this
                previous_list = self.scans_to_process
                self.scans_to_process = []
                timeout = time.time()+ 60*20 
                #QtTest.QTest.qWait(1000)

            if time.time() > timeout_for_live:
                break


class ScanListStream2(QThread):
    first_last_sid_sig = pyqtSignal(list)
    enableLiveButton = pyqtSignal(bool)
    def __init__(self,monitor_thread):
        super().__init__()
        self.scans_to_process = []
        self.monitor_thread = monitor_thread

    def run(self):
        first_sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
        self.enableLiveButton.emit(False)
        timeout = time.time() + 60*60   # 60 minute intervals
        timeout_for_live = time.time() + 60*60*24*5 # max 5 days active loop 
        previous_list = [100,200]

        while True:
            QtTest.QTest.qWait(1000*60)
            sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
            
            if sid-first_sid > 30:
                first_sid = int(sid-1)

            first_last_sid = [first_sid,sid]

            #if caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and not previous_list == first_last_sid and not self.monitor_thread.isRunning():
            if not previous_list == first_last_sid:
                    print(first_last_sid)
                    self.first_last_sid_sig.emit(first_last_sid) 
                    previous_list = first_last_sid
                    
            elif time.time() > timeout_for_live:
                break
                


class ScanNumberStream(QThread):
    scan_num = pyqtSignal(int)
    enableLiveButton = pyqtSignal(bool)
    def __init__(self, buffertime):
        super().__init__()
        self.buffertime = buffertime
    
    def run(self):
        self.enableLiveButton.emit(False)
        sid_sent = 100
        while True:
            QtTest.QTest.qWait(1000)
            sid = int(caget('XF:03IDC-ES{Status}ScanID-I'))
            hdr = db[int(sid)]
            start_doc = hdr["start"]
            if not start_doc["plan_type"] in ("FlyPlan1D",):
                if caget('XF:03IDC-ES{Status}ScanRunning-I') == 0 and not sid==sid_sent:
                
                    self.scan_num.emit(sid)
                    sid_sent = sid
                    logger.info(f"new scan signal sent: {sid}")
                    print(f"new scan signal sent: {sid}")
                    self.sleep(self.buffertime)
                    #QtTest.QTest.qWait(5000)
                
class scanStatus(QThread):
    scan_sts = pyqtSignal(int)
    scan_num = pyqtSignal(int)

    def run(self):
        while True:
            QtTest.QTest.qWait(2000)
            
            self.scan_num.emit(int(caget('XF:03IDC-ES{Status}ScanID-I')))
            self.scan_sts.emit(caget('XF:03IDC-ES{Status}ScanRunning-I'))

class XANESProcessing(QThread):
    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict

    def run(self):
        run_build_xanes_dict(self.paramDict)

class XANESBatchProcessing(QThread):

    current_process = pyqtSignal(dict)
    current_iter = pyqtSignal(int)

    def __init__(self, batch_job_dict):
        super().__init__()
        self.batch_job_dict = batch_job_dict
        self.paramDict = {}

    def run(self):
        n = 0 
        for key, value in self.batch_job_dict.items():

            save_dict = self.paramDict
            self.paramDict = value
            self.current_process.emit(self.paramDict)
            n = +1
            self.current_iter.emit(n)

            try:
                run_build_xanes_dict(self.paramDict)
                
                h = db[int(save_dict["first_sid"])]
                start_doc = h["start"]

                save_dict["n_points"] = (start_doc["num1"],start_doc["num2"])
                save_dict["exposure_time_sec"] = start_doc["exposure_time"]
                save_dict["step_size_um"] = start_doc["per_points"]

                outfile = os.path.join(f'{save_dict["cwd"]}/{save_dict["first_sid"]}-{save_dict["last_sid"]}',f'{save_dict["first_sid"]}_{save_dict["last_sid"]}.json')
                with open(outfile, "w") as fp:
                    json.dump(save_dict,fp, indent=6)
            except:
                pass

#using for track file mode
class Loadh5AndFitFromList(QThread):

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict
        #self.missed_scans = []
        self.scan_list_requested = self.paramDict["sidList"]
        self.failed_scans = []


    def run(self):

        for sid in self.scan_list_requested:
            fname = os.path.join(self.paramDict["wd"],f"scan2D_{int(sid)}.h5")
            print(f"{fname} exists")

            if not os.path.exists(fname):
                h = db[int(sid)]
                if bool(h.stop):
                    if h.start['plan_type'] != 'FlyPlan1D' or '1D_FLY_PANDA':
                        print(f"{sid = }, processing")
                        try:
                            make_hdf(int(sid),
                                    wd = self.paramDict["wd"],
                                    file_overwrite_existing = False,
                                    create_each_det = True,
                                    skip_scan_types = ['FlyPlan1D', '1D_FLY_PANDA']
                                    )
                            
                        except:
                            if sid not in self.failed_scans: 
                                self.failed_scans.append(sid)
                                #self.le_failed_scans.setText(self.failed_scans)
                            pass

                else:
                    print(f"{fname} exists; skipped")
        '''
        for sid in self.scan_list_requested:
            fitted_file_present = os.path.exists(os.path.join(self.paramDict["wd"],f"output_tiff_scan2D_{sid}"))
            h5_present = os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5"))
            if not fitted_file_present and h5_present:

                try:
                    pyxrf_batch(int(sid), 
                                int(sid), 
                                wd=self.paramDict["wd"], 
                                param_file_name=self.paramDict["xrfParam"], 
                                scaler_name=self.paramDict["norm"], 
                                save_tiff=self.paramDict["saveXRFTiff"],
                                save_txt = False,
                                ignore_datafile_metadata = True,
                                fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
                                quant_ref_eline = self.paramDict.get("quant_calib_elem",'')
                                )
    
                except : pass
        '''

        try:
            pyxrf_batch(int(self.scan_list_requested[0]), 
                        int(self.scan_list_requested[-1]), 
                        wd=self.paramDict["wd"], 
                        param_file_name=self.paramDict["xrfParam"], 
                        scaler_name=self.paramDict["norm"], 
                        save_tiff=self.paramDict["saveXRFTiff"],
                        save_txt = False,
                        ignore_datafile_metadata = True,
                        fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
                        quant_ref_eline = self.paramDict.get("quant_calib_elem",''),
                        interpolate_to_uniform_grid = self.paramDict.get("interpolate_to_uniform_grid",True)
                        )
            QtTest.QTest.qWait(5000)
    
        except : pass


        print(f"failed to process {self.failed_scans}")
                    
        # for sid in self.scan_list_requested:
        #     fit_fname = os.path.join(self.paramDict["wd"],f"output_tiff_scan2D_{sid}")
        #     if os.path.exists(fname) and self.paramDict["XRFfit"] and not os.path.exists(fit_fname):
        #         try:
        #             pyxrf_batch(int(sid), 
        #                         int(sid), 
        #                         wd=self.paramDict["wd"], 
        #                         param_file_name=self.paramDict["xrfParam"], 
        #                         scaler_name=self.paramDict["norm"], 
        #                         save_tiff=self.paramDict["saveXRFTiff"],
        #                         save_txt = False,
        #                         ignore_datafile_metadata = True,
        #                         fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
        #                         quant_ref_eline = self.paramDict.get("quant_calib_elem",'')
        #                         )
        #         except:
        #             if sid not in self.failed_scans: 
        #                 self.failed_scans.append(sid)
        #                 self.le_failed_scans.setText(self.failed_scans)
        #             pass
                        


class Loadh5AndFitFromListLive(QThread):

    skipped_1d_scans_sig = pyqtSignal(deque)

    def __init__(self, paramDict,skipped_1d):
        super().__init__()
        self.paramDict = paramDict
        #self.missed_scans = []
        self.scan_list_requested = self.paramDict["sidList"]
        self.skipped_1d = skipped_1d
        self.failed_scans = []

    def run(self):

        for sid in self.scan_list_requested:
            if not os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5")):
                h = db[int(sid)]
                if bool(h.stop):
                    if sid not in self.skipped_1d and h.start['plan_type'] != 'FlyPlan1D':
                        print(f"{sid = }, processing")
                        
                        try:
                            make_hdf(int(sid),
                                    wd = self.paramDict["wd"],
                                    file_overwrite_existing = True,
                                    create_each_det = True,
                                    skip_scan_types = ['FlyPlan1D','1D_FLY_PANDA']
                                    )
                            
                        except: 
                            if sid not in self.failed_scans: 
                                self.failed_scans.append(sid)
                                #self.le_failed_scans.setText(self.failed_scans)
                            
                            pass

                    else:
                       if sid not in self.skipped_1d:
                           self.skipped_1d.append(int(sid))

        
        for sid in self.scan_list_requested:
            fitted_file_present = os.path.exists(os.path.join(self.paramDict["wd"],f"output_tiff_scan2D_{sid}"))
            h5_present = os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5"))
            
            if not fitted_file_present and h5_present:
                try:
                    pyxrf_batch(int(sid), 
                                int(sid), 
                                wd=self.paramDict["wd"], 
                                param_file_name=self.paramDict["xrfParam"], 
                                scaler_name=self.paramDict["norm"], 
                                save_tiff=self.paramDict["saveXRFTiff"],
                                save_txt = False,
                                ignore_datafile_metadata = True,
                                fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
                                quant_ref_eline = self.paramDict.get("quant_calib_elem",''),
                                interpolate_to_uniform_grid = self.paramDict.get("interpolate_to_uniform_grid",True)
                                )
                    print(f"{self.paramDict['wd']}/output_tiff_scan2D_{sid} created")
                except: pass
        
        
        print(f"failed to process {self.failed_scans}")
        
        # for sid in self.scan_list_requested:    
        #     QtTest.QTest.qWait(500)
        #     fitted_file_present = os.path.exists(os.path.join(self.paramDict["wd"],f"output_tiff_scan2D_{sid}"))
        #     h5_present = os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5"))

        #     if self.paramDict["XRFfit"] and h5_present and not fitted_file_present:

        #         try:

        #             pyxrf_batch(int(sid), 
        #                         int(sid), 
        #                         wd=self.paramDict["wd"], 
        #                         param_file_name=self.paramDict["xrfParam"], 
        #                         scaler_name=self.paramDict["norm"], 
        #                         save_tiff=self.paramDict["saveXRFTiff"],
        #                         save_txt = False,
        #                         ignore_datafile_metadata = True,
        #                         fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
        #                         quant_ref_eline = self.paramDict.get("quant_calib_elem",'')
        #                         )
        #         except:
        #             if sid not in self.failed_scans: 
        #                 self.failed_scans.append(sid)
        #                 self.le_failed_scans.setText(self.failed_scans)
        #             pass
                            
        #print(f" emitted {self.skipped_1d = }")
        self.skipped_1d_scans_sig.emit(self.skipped_1d)               

class Loadh5AndFitForLive(QThread):
    
    h5loaded = pyqtSignal(int)
    last_processed = pyqtSignal(int)

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict
        
        self.scan_list_requested = self.paramDict["sidList"]

    def run(self):

        self.missed_scans = xrf_load_and_fit_from_list(self.scan_list_requested, 
                                                       self.paramDict)


        if self.missed_scans:
                            
            missed_twice = xrf_load_and_fit_from_list(self.missed_scans, 
                                                       self.paramDict)


def xrf_load_and_fit_from_list(sid_list, param_dict):
    
    missed_scans = []

    for sid in sid_list:
        
        try:

            make_hdf(int(sid),
                    wd = param_dict["wd"],
                    file_overwrite_existing = param_dict['file_overwrite_existing'],
                    create_each_det = True,
                    skip_scan_types = ['FlyPlan1D','1D_FLY_PANDA']
                    )
            
            QtTest.QTest.qWait(1000)
            if param_dict["XRFfit"]:
                
                pyxrf_batch(int(sid), 
                            int(sid), 
                            wd=param_dict["wd"], 
                            param_file_name=param_dict["xrfParam"], 
                            scaler_name=param_dict["norm"], 
                            save_tiff=param_dict["saveXRFTiff"],
                            save_txt = False,
                            ignore_datafile_metadata = True,
                            fln_quant_calib_data = param_dict.get("quant_calib_file",''),
                            quant_ref_eline = param_dict.get("quant_calib_elem",'',),
                            interpolate_to_uniform_grid = paramDict.get("interpolate_to_uniform_grid",True)
                            )
        except:
                missed_scans = missed_scans.append(sid)
                
        return missed_scans


class Loadh5AndFit(QThread):
    
    h5loaded = pyqtSignal(int)
    last_processed = pyqtSignal(int)

    def __init__(self, paramDict):
        super().__init__()
        self.paramDict = paramDict
        self.missed_scans = []
        self.failed_scans = []


    def run(self):
        logger.info("h5 thread started")
        QtTest.QTest.qWait(500)

        #print(f"{self.paramDict['file_overwrite_existing'] = }")
        print(f"\n Process: make hdf in batch-->xrf fitting in batch" )
        for sid in self.paramDict["sidList"]: #filter for 1d

            try:
                hdr = db[int(sid)]
                start_doc = hdr["start"]
                if not start_doc["plan_type"] in ("FlyPlan1D",):
                    print(f"Loading h5 data of {sid = }")

                make_hdf(
                    int(sid), 
                    wd = self.paramDict["wd"],
                    file_overwrite_existing = self.paramDict['file_overwrite_existing'],
                    create_each_det = True,
                    skip_scan_types = ['FlyPlan1D','1D_FLY_PANDA']
                    )
                
                print(f"Pyxrf h5 for {sid = } is created ")
                
            except:
                if sid not in self.failed_scans: 
                    self.failed_scans.append(sid)
                    #self.le_failed_scans.setText(self.failed_scans)
                pass
            QtTest.QTest.qWait(1000)


        try:
            print(f"interpolation marker: {self.paramDict.get('interpolate_to_uniform_grid',True)}")

            pyxrf_batch(int(self.paramDict["sidList"][0]), 
                        int(self.paramDict["sidList"][-1]), 
                        wd=self.paramDict["wd"], 
                        param_file_name=self.paramDict["xrfParam"], 
                        scaler_name=self.paramDict["norm"], 
                        save_tiff=self.paramDict["saveXRFTiff"],
                        save_txt = False,
                        ignore_datafile_metadata = True,
                        fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
                        quant_ref_eline = self.paramDict.get("quant_calib_elem",''),
                        interpolate_to_uniform_grid = self.paramDict.get("interpolate_to_uniform_grid",True)
                        )
            
            print(f"Batch fitting from {self.paramDict['sidList'][0]} to {self.paramDict['sidList'][-1]} is done")
        

        except Exception as e: print("Error: "+e)

        # for sid in self.paramDict["sidList"]:
        #     h5_present = os.path.exists(os.path.join(self.paramDict["wd"],f"scan2D_{sid}.h5"))
        #     fitted_file_present = os.path.exists(os.path.join(self.paramDict["wd"],f"output_tiff_scan2D_{sid}"))
        #     if self.paramDict["XRFfit"] and h5_present:
        #         try:
        #             pyxrf_batch(int(sid), 
        #                         int(sid), 
        #                         wd=self.paramDict["wd"], 
        #                         param_file_name=self.paramDict["xrfParam"], 
        #                         scaler_name=self.paramDict["norm"], 
        #                         save_tiff=self.paramDict["saveXRFTiff"],
        #                         save_txt = False,
        #                         ignore_datafile_metadata = True,
        #                         fln_quant_calib_data = self.paramDict.get("quant_calib_file",''),
        #                         quant_ref_eline = self.paramDict.get("quant_calib_elem",'')
        #                         )
            

                # except :
                #     if sid not in self.failed_scans: 
                #         self.failed_scans.append(sid)
                #         #self.le_failed_scans.setText(self.failed_scans)                       
                #     print(f" Fit not completed; {sid = }")
                #     self.missed_scans.append(sid)
                #     pass

        
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
            ignore_datafile_metadata = True,
            fln_quant_calib_data = self.paramDict.get("quant_calib_file", ''),
            quant_ref_eline = self.paramDict.get("quant_calib_elem", '')
            )
        
class TrackingFileToScanNumerThreadLive(QThread):

    scan_list_sig = pyqtSignal(list)

    def __init__(self, tracking_file_name):
        super().__init__()
        self.tracking_file_name = tracking_file_name
        self.previous_list = []
        self.repeated_attempt = 0

    def run(self):

        while True:

            if self.tracking_file_name.endswith(".csv"):
                log = pd.read_csv(self.tracking_file_name).dropna()
                if "scan_id" in log.columns:
                    self.scan_list = list(log['scan_id'].to_numpy(dtype = 'int'))
                else:
                    # Find columns starting with "scan" (case-insensitive)
                    matching_columns = [col for col in log.columns if col.lower().startswith("scan")]
                    if matching_columns:
                        print(f"No 'scan_id' found. Matching columns: {matching_columns}")
                        self.scan_list = list(log[str(matching_columns[0])])
                    else:
                        print("No matching columns found.")
            else:
                self.tracking_file_name.endswith(".txt")
                log = np.loadtxt(self.tracking_file_name, dtype = int)
                self.scan_list = list(log[:,0])



            if self.scan_list != self.previous_list: 
                self.scan_list_sig.emit(self.scan_list)
                self.previous_list = self.scan_list
                QtTest.QTest.qWait(30*1000)
                self.repeated_attempt = 0

            else:
                self.repeated_attempt += 1
                print(f"{self.repeated_attempt = }")
                QtTest.QTest.qWait(30*1000)

            if self.repeated_attempt >3600:
                break



        

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
    #print("app loading")
    window.show()
    sys.exit(app.exec_())
