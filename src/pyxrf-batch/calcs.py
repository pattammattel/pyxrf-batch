import sys, os, time, subprocess, logging, gc, h5py, traceback
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger()


def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))



def getEnergyNScalar(h='h5file'):
    """
    Function retrieve the ion chamber readings and
    mono energy from an h5 file created within pyxrf at HXN

    input: h5 file path
    output1: normalized IC3 reading ("float")
    output2: mono energy ("float")

    """
    # open the h5

    f = h5py.File(h, 'r')
    # get Io and IC3,  edges are removeed to exclude nany dropped frame or delayed reading
    Io = np.array(f['xrfmap/scalers/val'])[1:-1, 1:-1, 0].mean()
    I = np.array(f['xrfmap/scalers/val'])[1:-1, 1:-1, 2].mean()
    # get monoe
    mono_e = f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy']
    f.close()
    # return values
    return I / Io, mono_e

def getCalibSpectrum(path_=os.getcwd()):
    """
	Get the I/Io and enegry value from all the h5 files in the given folder

	-------------
	input: path to folder (string), if none, use current working directory
	output: calibration array containing energy in column and log(I/Io) in the other (np.array)
    -------------
	"""

    # get the all the files in the directory
    fileList = list(absoluteFilePaths(path_))

    # empty list to add values
    spectrum = []
    energyList = []

    for file in sorted(fileList):
        if file.endswith('.h5'):  # filer for h5
            IbyIo, mono_e = getEnergyNScalar(h=file)
            energyList.append(mono_e)
            spectrum.append(IbyIo)

    # get the output in two column format
    calib_spectrum = np.column_stack([energyList, (-1 * np.log10(spectrum))])
    # sort by energy
    calib_spectrum = calib_spectrum[np.argsort(calib_spectrum[:, 0])]
    # save as txt to the parent folder
    #np.savetxt('calibration_spectrum.txt', calib_spectrum)
    logger.info("calibration spectrum saved in : {path_} ")

    return calib_spectrum