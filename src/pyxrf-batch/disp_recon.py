import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import tifffile as tf

space = 15
def save_recon(sid_s,sid_e,interval, work_space = '/data/home/hyan/recon/scan_100000', folder_name = 't1', background_removed=False):
    sid_list = range(sid_s, sid_e+1, interval)
    num = np.size(sid_list)
    for i in range(num):
        sid = sid_list[i]
        sid_str = '{}'.format(sid)
        '''
        obj_file = ''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_data/', 'recon_', sid_str, '_', folder_name, '_object.npy'])
        prb_file = ''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_data/', 'recon_', sid_str, '_', folder_name, '_probe.npy'])
        obj = np.load(obj_file)
        prb = np.load(prb_file)
        prb_sz = np.shape(prb)
        obj_sz = np.shape(obj)
        obj_c = obj[prb_sz[0]/2+space:obj_sz[0]-prb_sz[0]/2-space, prb_sz[1]/2+space:obj_sz[1]-prb_sz[1]/2-space]
        '''
        prb, obj_c = read_recon(sid, work_space=work_space, folder_name = folder_name,background_removed=background_removed)
        obj_c_arg = np.angle(obj_c)
        obj_c_amp = np.abs(obj_c)
        
        '''
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12.8, 4.8))
        im0 = ax0.imshow(obj_c_amp,cmap='gist_gray',aspect='equal')
        fig.colorbar(im0, ax=ax0)
        ax0.set_title('obj amp')
        im1 = ax1.imshow(obj_c_arg,cmap='gist_gray',aspect='equal')
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('obj arg')
        fig.savefig(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/obj_scanned_area.tif']))
        plt.close()
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12.8, 4.8))
        im0 = ax0.imshow(np.abs(prb),cmap='jet',aspect='equal')
        fig.colorbar(im0, ax=ax0)
        ax0.set_title('prb amp')
        im1 = ax1.imshow(np.angle(prb),cmap='jet',aspect='equal')
        fig.colorbar(im1, ax=ax1)
        ax1.set_title('prb arg')
        '''
        fig = create_recon_fig(prb, obj_c)

        fig.savefig(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_result.tif']))
        plt.close()
        np.savetxt(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/obj_c_arg.txt']),obj_c_arg)
        np.savetxt(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/obj_c_amp.txt']),obj_c_amp)
        
        np.savetxt(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/prb_arg.txt']),np.angle(prb))
        np.savetxt(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/prb_amp.txt']),np.abs(prb))
        
        tf.imsave(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/obj_c_arg.tiff']),obj_c_arg, imagej = True)
        tf.imsave(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/obj_c_amp.tiff']),obj_c_amp, imagej = True)
        
        tf.imsave(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/prb_arg.tiff']),np.angle(prb), imagej = True)
        tf.imsave(''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/prb_amp.tiff']),np.abs(prb), imagej = True)

        #plt.show() 
def disp_recon(sid, work_space = '/data/home/hyan/recon/scan_100000', folder_name = 't1',background_removed=False):
    '''
    sid_str = '{}'.format(sid)
    obj_file = ''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_data/', 'recon_', sid_str, '_', folder_name, '_object.npy'])
    prb_file = ''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_data/', 'recon_', sid_str, '_', folder_name, '_probe.npy'])
    obj = np.load(obj_file)
    prb = np.load(prb_file)
    obj = np.fliplr(np.rot90(obj))
    prb = np.fliplr(np.rot90(prb))
    prb_sz = np.shape(prb)
    obj_sz = np.shape(obj)
    obj_c = obj[prb_sz[0]/2+space:obj_sz[0]-prb_sz[0]/2-space, prb_sz[1]/2+space:obj_sz[1]-prb_sz[1]/2-space]
    '''
    prb, obj_c = read_recon(sid, work_space=work_space, folder_name = folder_name, background_removed=background_removed)
    '''
    obj_c_arg = np.angle(obj_c)
    obj_c_amp = np.abs(obj_c)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9.6, 7.2))
    im0 = axs[0,0].imshow(np.abs(prb),cmap='jet',aspect='auto')
    fig.colorbar(im0, ax=axs[0,0])
    axs[0,0].set_title('prb amp')
    im1 = axs[0,1].imshow(np.angle(prb),cmap='jet',aspect='auto')
    fig.colorbar(im1, ax=axs[0,1])
    axs[0,1].set_title('prb arg')      

    im2 = axs[1,0].imshow(obj_c_amp,cmap='bone',aspect='auto')
    fig.colorbar(im2, ax=axs[1,0])
    axs[1,0].set_title('obj amp')
    im3 = axs[1,1].imshow(obj_c_arg,cmap='bone',aspect='auto')
    fig.colorbar(im3, ax=axs[1,1])
    axs[1,1].set_title('obj arg')
    '''
    create_recon_fig(prb, obj_c)
    plt.show()
def create_recon_fig(prb, obj_c):
    obj_c_arg = np.angle(obj_c)
    obj_c_amp = np.abs(obj_c)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9.6, 7.2))
    im0 = axs[0,0].imshow(np.abs(prb),cmap='jet',aspect='auto')
    fig.colorbar(im0, ax=axs[0,0])
    axs[0,0].set_title('prb amp')
    im1 = axs[0,1].imshow(np.angle(prb),cmap='jet',aspect='auto')
    fig.colorbar(im1, ax=axs[0,1])
    axs[0,1].set_title('prb arg')

    im2 = axs[1,0].imshow(obj_c_amp,cmap='bone',aspect='auto')
    fig.colorbar(im2, ax=axs[1,0])
    axs[1,0].set_title('obj amp')
    im3 = axs[1,1].imshow(obj_c_arg,cmap='bone',aspect='auto')
    fig.colorbar(im3, ax=axs[1,1])
    axs[1,1].set_title('obj arg')
    
    return fig

def read_recon(sid, work_space, folder_name, background_removed):
    sid_str = '{}'.format(sid)
    obj_file = ''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_data/', 'recon_', sid_str, '_', folder_name, '_object.npy'])
    prb_file = ''.join([work_space, '/recon_result', '/S', sid_str, '/', folder_name, '/recon_data/', 'recon_', sid_str, '_', folder_name, '_probe.npy'])
    obj = np.load(obj_file)
    prb = np.load(prb_file)
    obj = np.fliplr(np.rot90(obj))
    prb = np.fliplr(np.rot90(prb))
    prb_sz = np.shape(prb)
    obj_sz = np.shape(obj)
    obj_c = obj[np.int(prb_sz[0]/2)+space:obj_sz[0]-np.int(prb_sz[0]/2)-space, np.int(prb_sz[1]/2)+space:obj_sz[1]-np.int(prb_sz[1]/2)-space]
    obj_c_arg = np.angle(obj_c)
    obj_c_amp = np.abs(obj_c)
    if background_removed:
        obj_c_arg = remove_background(obj_c_arg)
    return prb, obj_c_amp*np.exp((0+1j)*obj_c_arg)

    
def fit_func(p0,px1,py1,):
    return lambda x,y:p0+x*px1+py1*y

def remove_background(im):
    #im_sz = np.shape(im)
    
    params = [0,0,0]
    err = lambda p: np.ravel(fit_func(*p)(*np.indices(im.shape)) - im)
    p, success = leastsq(err, params)
   
    return im - fit_func(*p)(*np.indices(im.shape))
