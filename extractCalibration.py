import numpy as np
import h5py
import tifffile as tf
import os



def getEnergyNScalar(h='h5file'):

    f = h5py.File(h, 'r')
    Io = np.array(f['xrfmap/scalers/val'])[:, :, 0].mean()
    I = np.array(f['xrfmap/scalers/val'])[:, :, 2].mean()
    mono_e = f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy']
    return I/Io, mono_e
    
def getCalibSpectrum(path_):
    
    fileList = os.listdir(path_)
    
    spectrum = []
    energyList = []
    
    for file in sorted(fileList):
        if file.endswith('.h5'):
            IbyIo, mono_e = getEnergyNScalar(h=file)
            energyList.append(mono_e)
            spectrum.append(IbyIo)
    
    calib_spectrum = np.column_stack([energyList,(-1*np.log10(spectrum))])
                
    return calib_spectrum
            
