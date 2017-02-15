__author__ = 'qiwang'
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer

def patch_quantize_01(P):
    assert np.all(P>=0.0) and np.all(P<=1.0)
    P = P*108.0
    P = np.round(P)
    P[(P>0.0) & (P<=10.5)] = 0.0
    P[(P>10.5) & (P<21.0)] = 21.0
    P = P/108.0
    return P
def patch_thresholding(P,thres_min=0.0,thres_max=1.0):
    P[P>thres_max] = thres_max
    P[P<thres_min] = thres_min
    return P
def show_samples(generator,Noise_Dim,data_obj,filename):
    if data_obj.pitch_scale:
        pitch_max = 1.0
    else:
        pitch_max = 108.0
    rows = 4
    sample_cols = 5
    input_noise = np.random.uniform(-1.0,1.0,(rows*sample_cols, Noise_Dim))
    samples = generator.predict(input_noise)
    topo_samples = samples.reshape(samples.shape[0],4,samples.shape[-1]/4)
    #get topological_view

    pv = PatchViewer(grid_shape=(rows,sample_cols + 1),patch_shape=(4,samples.shape[-1]/4), \
                     is_color=False)
    X = np.concatenate((data_obj.X_train,data_obj.X_val,data_obj.X_test),axis = 0)
    topo_X = X
    print('Shape of dataset is {}').format(X.shape)
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])

    for i in xrange(topo_samples.shape[0]):
        topo_sample = patch_quantize_01(patch_thresholding(topo_samples[i,:]/pitch_max))
        pv.add_patch(topo_sample * 2. -1.,rescale=False)
        if(i + 1) % sample_cols ==0:
            sample = samples[i,:]
            dists = np.square(X - sample).sum(axis = 1)
            j = np.argmin(dists)
            match = patch_quantize_01(patch_thresholding(topo_X[j,:]/pitch_max))
            pv.add_patch(match*2-1,rescale=False,activation = 1)
    print "Saving %s ..."%filename
    pv.save(filename)

#define show_sample_pairs
def show_sample_pairs(generator,Noise_Dim,data_obj,filename):
    if data_obj.pitch_scale:
        pitch_max = 1.0
    else:
        pitch_max = 108.0
    grid_shape = None

    input_noise = np.random.uniform(-1.0,1.0,(100, Noise_Dim))
    samples = generator.predict(input_noise)
    grid_shape = (10,20)
    matched = np.zeros((samples.shape[0] *2, samples.shape[1]))
    X = np.concatenate((data_obj.X_train,data_obj.X_val,data_obj.X_test),axis=0)
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    for i in xrange(samples.shape[0]):
        matched[2*i,:] = samples[i,:].copy()
        dists = np.square(X - samples[i,:]).sum(axis = 1)
        j = np.argmin(dists)
        matched[2*i+1,:] = X[j,:]
    samples = matched
    is_color = False

    samples = patch_quantize_01(patch_thresholding(samples/pitch_max))
    samples = samples * 2.0 - 1.0
    viewer = make_viewer(samples, grid_shape=grid_shape,patch_shape=(4,samples.shape[-1]/4),\
                          is_color=is_color,rescale=False)
    print "Saving %s ..."%filename
    viewer.save(filename)

#define a function for plotting the objective curve

def plot_objectives(loss_history,filename,figsize=(5,7),ms =5,fs=10,\
                    ls=10,ts=12,isepoch=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(211)
    loss = loss_history['train_d_loss']
    x = range(len(loss))
    ax.plot(x,loss,label = 'train_objective_d (%.2f)'%loss[-1],color = 'red')
    loss = loss_history['train_g_loss']
    ax.plot(x,loss,label = 'train_objective_g (%.2f)'%loss[-1], color = 'green')
    ax.legend(prop={'size':ls})
    ax.set_xlabel('epoch',fontsize = fs)
    ax.set_ylabel('obj',fontsize = fs)

    ax = fig.add_subplot(212)
    loss = loss_history['valid_d_loss']
    x = range(len(loss))
    ax.plot(x,loss,label = 'valid_objective_d (%.2f)'%loss[-1],color = 'red')
    loss = loss_history['valid_g_loss']
    ax.plot(x,loss,label = 'valid_objective_g (%.2f)'%loss[-1], color = 'green')
    ax.legend(prop={'size':ls})
    ax.set_xlabel('epoch',fontsize = fs)
    ax.set_ylabel('obj',fontsize = fs)

    print('Saving %s ...')%filename
    plt.savefig(filename)
    plt.close(fig)





