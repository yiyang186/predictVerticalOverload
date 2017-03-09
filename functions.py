# -*- coding:utf-8 -*-

import pandas as pd
import os
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from collections import OrderedDict
# %matplotlib inline

params = OrderedDict({'HEAD_MAG':1,'ALT_STD':1,'RALTC':1,'IAS':1,'MACH':1,'GS':1,
                      'WIN_SPDR':1, 'WIN_DIR':1,'DRIFT':1,'IVVR':1,'VRTG':8,'LATG':4,'LONG':4,'N11':1,'N12':1,'N21':1,
                      'N22':1,'EGT1':1,'EGT2':1,'TAT':1,'V2':1,'VAPP':1,'TLA1':1,'TLA2':1,'HPC_TMP1':1,'HPC_TMP2':1,
                      'OIL_PRS1':1,'OIL_PRS2':1,'OIL_QTY1':1, 'OIL_QTY2':1,'OIL_TMP1':1,'OIL_TMP2':1,'YAW_TRM':1,
                      'FF1':1,'FF2':1,'FLAPC':1,'AIL_OUBL':1,'AIL_OUBR':1,'SPD_BRK_SEL':1,'SPOIL_L2':1,
                      'SPOIL_L3':1,'SPOIL_L4':1,'SPOIL_L5':1,'RUDD':2,'RUD_PDL':2,'RUDD_FORCE':1,'ELEVL':2,'ELEVR':2,
                      'AOAL':1,'AOAR':1,'PITCH':4,'PITCH_CPT':4,'ROLL':2,'ROLL_CPT':4,'ALT_SEL':1,'MACH_SEL':1,
                      'MACH_SEL.1':1,'IVV_SEL':1,'SEL_CRS_VOR1':1,'SEL_CRS_VOR2':1, 'AIL_INBL':2, 'AIL_INBR':2,
                      'AIL_OUBL.1':1, 'AIL_OUBR.1':1,'DIST0':1,'ELEVL.1':2, 'ELEVR.1':2,'EPR_ENG1_ACTUA':1,
                      'EPR_ENG2_ACTUA':1,'EST_SID_SLIP':1, 'EW_SPDR':1})
workdir2 = "/home/pyy/data/cast/all_-60_15/"
bins = 10

def handle_filename(fn):
    fn.append('-'.join(fn))
    fn[4] = fn[4][:2] + ':' + fn[4][2:4] + ':' + fn[4][4:6] #着陆时间
    fn[3] = '20' + fn[3][4:] + '-' + fn[3][2:4] + '-' + fn[3][0:2]  #原来的着陆日期按“月/日/年”排列
    fn.append(fn[3] + ' ' + fn[4])
    return fn

def lookonevrtg(index_list):
    titles = ""
    for index in index_list:
        my_vrtg = df.ix[index][8:]
        my_vrtg.plot(figsize=(20, 5))
        titles += '|' + str(index)
    plt.axhline(0, c='black')
    plt.axvline(480, c='black')
    plt.xticks(range(0, my_vrtg.shape[0], 80),  ['{0}s'.format(x / 8 - 60) for x in range(0, my_vrtg.shape[0], 80)])
    plt.ylim(-0.25, 0.6)
    plt.title("FILE_NO="+titles + '|')
#plt.fill_between(np.arange(my_vrtg.shape[0]), my_vrtg-my_std, my_vrtg+my_std, color='b', alpha=0.2)

def composite_wind(win_spd, win_dir, head_mag):
    if win_spd.shape != win_dir.shape:
        return None
    t = np.pi / 180
    win_y = win_spd * np.sin((win_dir - head_mag) * t)
    win_x =  win_spd * np.cos((win_dir - head_mag) * t)
    return  win_x, win_y

def plot_var(matrix, max_vrtg, steps, bins, colname, draw=True, isdiff=False, outdir=None):
    if isdiff:
        print "diff"
        y = np.diff(matrix, n=1, axis=1).var(axis=1)
    else:
        y = matrix.var(axis=1)

    r1 = partical_plot_var_sum(y, max_vrtg, steps, colname, draw, isdiff, outdir=outdir)
    plt.figure()
    r2 = partical_plot_var_ave(y, max_vrtg, bins, colname, draw, isdiff, outdir=outdir)
    return list(r1), list(r2)

def import_data(workdir, df, colname, rate, iscorrt, t=75):
    size = rate * t
    
    matrix = np.zeros([df.shape[0], size])
    todelete = []
    for i, index in enumerate(df.index):
        fn = workdir + df.loc[index, "filename"]
        if os.path.exists(fn):
            fn_df = pd.read_csv(fn, usecols=[colname])
        else:
            todelete.append(i)
            continue
        if iscorrt:
            matrix[i, :] = fn_df[colname].dropna().values[:size] - fn_df[colname].dropna().values[-5 * rate:].mean()
        else:
            matrix[i, :] = fn_df[colname].dropna().values[:size]
    return np.delete(matrix, todelete, 0)

def partical_plot_var_sum(y, v, step, colname, draw=True, isdiff=False, outdir=None):
    y_up = y.shape[0] / step * step
    nrow = y.shape[0] / step
    y = y[: y_up].reshape((nrow, step))
    v = v[: y_up].reshape((nrow, step))
    y_sums =  y.sum(axis=1)
    y_sums1 = np.zeros(y_sums.shape)
    
    if max(y_sums) == 0:
        pivot = 0.00000001
    else:
        i = 0
        while(y_sums[i] == 0):
            i += 1
        pivot = y_sums[i]
    for i in range(y_sums.shape[0]):
        y_sums1[i] = y_sums[i] / pivot
    
    if draw:
        fig = plt.figure(figsize= (10, 7))
        plt.xticks(fontsize=10, rotation=45)
        plt.xlabel('Flights range', fontsize=14)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.scatter(np.arange(nrow)+1, y_sums1, c='red', s=50)
        ax1.plot(np.arange(nrow)+1, y_sums1)
        ax1.set_ylabel('Partical summative variance of {0} / times'.format(colname + ('_diff' if isdiff else '')), fontsize=14)
        
        bias = (y_sums1.max() - y_sums1.min()) * 0.05
        for i in range(nrow):
            ax1.text(i+0.8, y_sums1[i]+bias, format(y_sums[i], '.2e'), fontsize=10)

        ax2.violinplot(v.T, showmeans=False, showmedians=True)
        ax2.set_ylabel('VRTG(corrected)', fontsize=14)

        plt.xlim(0, nrow+1)
        plt.xticks(np.arange(nrow)+1, ['{0}-{1}'.format(x*step, (x+1)*step) for x in np.arange(nrow)])
        plt.title('Partical summative variance by flights({0}s~{1}s)'.format(t1, t2), fontsize=20)
        fig.savefig(u'{0}{1}_1.png'.format(outdir, colname+('_diff' if isdiff else u'')), dpi=75)
    return y_sums1

def partical_plot_var_ave(y, v, bins, colname, draw=True, isdiff=False, outdir=None):
    
    ns, bs = np.histogram(v, bins=bins, normed=False)
    aves = np.zeros((3, bins))
    for i in range(bins):
        aves[0, i] = y[np.logical_and(v > bs[i],  v <= bs[i+1])].mean()#sum() / ns[i]
        #aves[2, i] = y[np.logical_and(v > bs[i],  v <= bs[i+1])].std() * 1.96
    
    if aves[0,:].max() == 0:
        pivot = 0.0000001
    else:
        i = 0
        while(aves[0,i] == 0):
            i += 1
        pivot = aves[0,i]
    for i in range(bins):
        aves[1, i] = aves[0, i] / pivot
        #aves[2, i] = aves[2, i] / pivot
    
    if draw:
        fig = plt.figure(figsize= (10, 7))
        plt.xticks(fontsize=10, rotation=45)
        plt.xlabel('Range of VRTG', fontsize=14)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Flights", fontsize=14)
        ax2.bar(np.arange(bins)+0.6, ns, color='y', alpha=0.3)
        for i in range(bins):
            ax2.text(i+0.8, ns[i]+50, str(ns[i]), fontsize=10)

        ax1.scatter(np.arange(bins)+1, aves[1,:], c='red', s=50)
        ax1.plot(np.arange(bins)+1, aves[1,:])
        #ax1.fill_between(np.arange(bins)+1, aves[1,:]-aves[2,:], aves[1,:]+aves[2,:], color='b', alpha=0.2)
        
        bias = (aves[1, :].max() - aves[1, :].min()) * 0.05
        for i in range(bins):
            ax1.text(i+0.8, aves[1, i]+bias, format(aves[0, i], '.2e'), fontsize=10)
        ax1.set_ylabel('Partical average variance of {0} / times'.format(colname+ ('_diff' if isdiff else u'')), fontsize=14)

        plt.xlim(0, bins+1)
        plt.xticks(np.arange(bins)+1, ['{0}-{1}'.format(round(bs[i], 3), round(bs[i+1], 3)) for i in range(bins)])
        plt.title('Partical average variance by MAX_VRTG({0}s~{1}s)'.format(t1, t2), fontsize=20)
        fig.savefig(u'{0}{1}_2.png'.format(outdir, colname+('_diff' if isdiff else '')), dpi=75)
    return aves[1, :]

def partical_plot_var_ave_win(y, yp, yn, v, vp, vn, bins, colname, draw=True, isdiff=False, outdir=None):
    if isdiff:
        y = np.diff(y, n=1, axis=1).var(axis=1)
        yp = np.diff(yp, n=1, axis=1).var(axis=1)
        yn = np.diff(yn, n=1, axis=1).var(axis=1)
    else:
        y = y.var(axis=1)
        yp = yp.var(axis=1)
        yn = yn.var(axis=1)
    
    ns, bs = np.histogram(v, bins=bins, normed=False)
    aves = np.zeros([6, bins])
    for i in range(bins):
        aves[0, i] = y[np.logical_and(v > bs[i],  v <= bs[i+1])].mean()
        aves[2, i] = yp[np.logical_and(vp > bs[i],  vp <= bs[i+1])].mean()
        aves[4, i] = yn[np.logical_and(vn > bs[i],  vn <= bs[i+1])].mean()
    
    pivot = aves[0,0]
    aves[1,:] = aves[0, :] / pivot
    aves[3, :] = aves[2, :] / pivot
    aves[5, :] = aves[4, :] / pivot
    mask = np.isfinite(aves)
    
    if draw:
        fig = plt.figure(figsize= (10, 7))
        plt.xticks(fontsize=10, rotation=45)
        plt.xlabel('Range of VRTG', fontsize=14)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Flights", fontsize=14)
        ax2.bar(np.arange(bins)+0.6, ns, color='y', alpha=0.3)
        for i in range(bins):
            ax2.text(i+0.8, ns[i]+50, str(ns[i]), fontsize=10)

        ax1.plot((np.arange(bins)+1)[mask[1,:]], aves[1,:][mask[1,:]], c='red', label='full', linestyle='-', marker='o')
        ax1.plot((np.arange(bins)+1)[mask[3,:]], aves[3,:][mask[3,:]], c='green', label='positive', linestyle='-', marker='*')
        ax1.plot((np.arange(bins)+1)[mask[5,:]], aves[5,:][mask[5,:]], c='blue', label='negative', linestyle='-', marker='^')
        #ax1.fill_between(np.arange(bins)+1, aves[1,:]-aves[2,:], aves[1,:]+aves[2,:], color='b', alpha=0.2)
        
#         bias = (aves[1, :].max() - aves[1, :].min()) * 0.05
#         for i in range(bins):
#             ax1.text(i+0.8, aves[1, i]+bias, format(aves[0, i], '.2e'), fontsize=10)
        ax1.set_ylabel('Partical average variance of {0} / times'.format(colname+ ('_diff' if isdiff else u'')), fontsize=14)
        ax1.set_ylim(0, 4)
        plt.xlim(0, bins+1)
        plt.xticks(np.arange(bins)+1, ['{0}-{1}'.format(round(bs[i], 3), round(bs[i+1], 3)) for i in range(bins)])
        plt.title(colname, fontsize=20)
        ax1.legend(loc="upper left")
        #fig.savefig(u'{0}{1}_2.png'.format(outdir, colname+('_diff' if isdiff else '')), dpi=75)
    return aves

import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title=u'混淆矩阵', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(2)
    thresh = cm.sum() * 0.25
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Real')
    plt.xlabel('Predict')