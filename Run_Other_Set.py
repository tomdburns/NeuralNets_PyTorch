"""
Runs the 'Other' set for each subset through the NN model
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from nets.nets import *
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

COLS    = 'b0_c,d0_c,q1_c,q2_c,U1_c,U2_c,b0_n,d0_n,q1_n,q2_n,U1_n,U2_n,StructuredDensity,InletTemp,tads,tblow,tevac,Pint,Plow,v0'.split(',')
device  = torch.device('cpu')


#TARGET = 'Purity'
TARGET = 'EPSA'


def main():
    """main"""
    model_file = 'Models\\NN_Model_4Layers_%s_0.pkl' % TARGET

    hps       = pkl.load(open('%s Model\\%s_4Layer_NNHyperParameters.pkl' % (TARGET, TARGET), 'rb'))
    net = Net4(nhidden1=hps[1], nhidden2=hps[2], nhidden3=hps[3],
               nhidden4=hps[4], dp_prob=hps[5], nndevice=device)
    net.load_state_dict(torch.load('Models\\NN_Model_4Layers_Targ_%s_0.pkl' % TARGET, map_location=device))
    scaler    = pkl.load(open('%s Model\\Scaler_%s_N4.pkl' % (TARGET, TARGET), 'rb'))
    logoffs   = pkl.load(open('%s Model\\LogOffsets.pkl' % TARGET, 'rb'))

    data = pd.read_csv('Subsets\\%s_Other_subset.csv' % TARGET)
    arr  = np.array(data[COLS])
    act  = np.array(data[TARGET])

    # Now that we've built the array we need to transform the data into model readable format
    # This starts with logging and offsetting specific values
    # Step 1 - Identify logged values
    log_ids = []
    for i, col in enumerate(COLS):
        if col in logoffs:
            log_ids.append(i)

    # Now perform Log transform operations to those columns
    for id in log_ids:
        col = COLS[id]
        if logoffs[col][0]:
            arr[:, id] = np.log10(arr[:, id] + logoffs[col][1])
        else:
            if min(arr[:, id]) > 0.:
                arr[:, id] = np.log10(arr[:, id])
            else:
                new = []
                for valu in arr[:, id]:
                    if valu <= 0:
                        new.append(logoffs[col][1])
                    else:
                        new.append(valu)
                arr[:, id] = np.array(new)
                arr[:, id] = np.log10(arr[:, id])

    # Finally, scale the data and return it to the parent function
    #_arr = scaler.transform(_arr)
    try:
        arr = scaler.transform(arr)
    except ValueError:
        print(arr)
        print('Value Error in Scaler Transformation')
        exit()

    # Convert to torch tensors
    arr = torch.from_numpy(arr).float()
    arr = arr.to(device)

    net.eval()
    pred = net(arr)
    pred = np.array([y[0] for y in pred.detach().numpy()])

    npred = []
    for val in pred:
        if 'Purity' == TARGET or 'Recovery' == TARGET:
            npred.append(min(val, 100.))
        else:
            npred.append(val)
    npred = np.array(npred)

    lo = min(min(npred), min(act))
    hi = max(max(npred), max(act))

    r, p = pearsonr(act, npred)
    r2   = r**2

    plt.hexbin(act, npred, cmap='jet', mincnt=1, bins='log')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('%s (R$^2$ = %.2f)' % (TARGET, r2))
    plt.plot([lo, hi], [lo, hi], color='k', linestyle='--')
    plt.show()


if __name__ in '__main__':
    main()
