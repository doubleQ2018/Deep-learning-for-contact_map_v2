#!/usr/bin/env python

import numpy as np

def accuracy(y_out, y, k):
    L = y.shape[-2]
    top_k = L/k
    y_out = y_out[:,:,:,1:]
    #y = y[:,:,:,1:] ### when input shape of y is (, L, L, 2)
    y_flat = y_out.flatten()
    y = y.flatten()
    indeces = [[index, val] for index, val in enumerate(y_flat)]
    indeces = sorted(indeces, key=lambda x: x[1], reverse=True)
    right = 0

    # generate predicted matrix
    pred_matrix = np.zeros((L, L))

    for i in range(top_k):
        cursor = indeces[i][0]
        right += y[cursor]
        pred_matrix[cursor/L][cursor%L] = 1
    return 1.0 * right / top_k, pred_matrix

def topKaccuracy(y_out, y, k):
    
    y_out = y_out[0]
    y = y[0]
    
    L = y.shape[-2]

    m = np.ones_like(y, dtype=np.int8)
    lm = np.triu(m, 24)
    mm = np.triu(m, 12)
    sm = np.triu(m, 6)

    avg_pred = (y_out + y_out.transpose((1, 0, 2))) / 2.0

    truth = np.concatenate((avg_pred, y[..., np.newaxis]), axis=-1)

    accs = []

    for x in [lm, mm, sm]:
        selected_truth = truth[x.nonzero()]

        selected_truth_sorted = selected_truth[(selected_truth[:, 1]).argsort()[::-1]]

        tops_num = min(selected_truth_sorted.shape[0], L/k)

        truth_in_pred = selected_truth_sorted[:, 2].astype(np.int8)

        corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
        
        acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)

        accs.append(acc)

    return accs

def topKprediction(y_out, k):
    y_out = y_out[0]

    return 0


def one_hot(contact_map):
    # change the shape to (L, L, 2) 
    tmp = np.where(contact_map > 0, 0, 1)
    true_contact = np.stack((tmp, contact_map), axis=-1)
    return true_contact

def weights_output(true_contact):
    L = true_contact.shape[0]
    sums = L * L
    ones = true_contact.sum()
    zeros = sums - ones
    one_weight = np.ones((L, L)) * (0.1/ones)
    zero_weight = np.ones((L, L)) * (0.9/zeros)
    weights = np.stack((zero_weight, one_weight), axis=-1)
    tmp = np.where(true_contact>0, 0, 1)
    true_contact = np.stack((tmp, true_contact), axis=-1)
    
    return weights * true_contact
