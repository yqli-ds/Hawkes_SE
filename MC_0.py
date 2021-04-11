import numpy as np
import scipy
import copy
import glob
import scipy.io as sio

import pandas as pd
import time
import csv
import os
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import poisson, norm, gamma, bernoulli

from utility_v1 import *

if __name__ == '__main__':

    pathss = ''
    pathss_save = ''
    filess = glob.glob(pathss + '*.txt')

    file_i = filess[0]
    data = (pd.read_csv(file_i, sep=" ", header=None).values).astype(float)
    name = file_i[-10:-4]
    ids = np.random.rand()

    NumEdge = 10000
    # preprocess data
    data = data[:NumEdge]
    # not consider the interaction to myself
    data = data[(data[:, 0]!=data[:, 1])]
    # unique_id = all sender and receiver id, sorted increasing
    unique_id = np.unique(data[:, :2]) #np.sort(np.unique(data[:, :2]))
    # data_id = first two column of data, id of sender and receiver
    data_id = data[:, :2]
    # change all the id to the range of 0-len(unique_id)
    for ii in range(len(unique_id)):
        data_id[data_id == unique_id[ii]] = ii
    # normalize time
    data[:, 2] = (data[:, 2] / (24 * 3600.0))
    data[:, 2] = data[:, 2] - np.min(data[:, 2])

    KK = 10
    dataNum = len(unique_id)

    nodepair = data[:, :2].astype(int)
    eventtime = data[:, 2]

    index = np.argsort(eventtime)
    eventtime = np.asarray(eventtime)[index]
    nodepair = np.asarray(nodepair)[index]


    time_inteval_receive_j_list, time_inteval_non_receive_j_list, sending_time_list, sender_receiver_num, \
    mutually_exciting_pair, receiving_j_list, time_inteval_receive_j_to_i_list, sending_time_list, pis_list, \
    pis0, C_i_list, b_ij, betas, a_betas, b_betas, Lambdas, a_Lambda, b_Lambda, M, a_M, b_M, alpha, a_alpha, \
    b_alpha, Delta_pis, Taus_kij, last_time_inteval_non_receive_j_list, last_time_inteval_receive_j_list, \
    last_time_inteval_receive_j_to_i_list = initialize_model(dataNum, KK, nodepair, eventtime)

    #### Iteration
    IterationTime = 200
    sCTEM = sCTEM_class(dataNum, KK, pis_list, C_i_list, time_inteval_receive_j_list,
                        time_inteval_non_receive_j_list, b_ij, betas, a_betas, b_betas, Lambdas, a_Lambda, b_Lambda, \
                        M, a_M, b_M, alpha, a_alpha,b_alpha, Delta_pis, \
                        Taus_kij, sending_time_list, sender_receiver_num, mutually_exciting_pair, receiving_j_list, \
                        time_inteval_receive_j_to_i_list, last_time_inteval_non_receive_j_list, \
                        last_time_inteval_receive_j_list, last_time_inteval_receive_j_to_i_list)

    auc_seq = []
    time_seq = []
    test_precision_seq = []


    sCTEM.sample_b_ij(nodepair, eventtime)
    sCTEM.sample_u_ij(nodepair, eventtime)

    mean_Lambdas = 0
    mean_betas = 0
    mean_pi = []
    mean_C = []
    seq_delta = []
    seq_tau = []
    seq_alpha = []
    running_time = []
    for ci in range(len(sCTEM.pis_list)):
        ci_pi = []
        ci_CC = []
        for di in range(len(sCTEM.pis_list[ci])):
            ci_pi.append(np.zeros(sCTEM.pis_list[ci][di].shape))
            ci_CC.append(np.zeros(sCTEM.C_i_list[ci][di].shape))
        mean_pi.append(ci_pi)
        mean_C.append(ci_CC)

    burnInTime = int(IterationTime/2)
    for ite in range(IterationTime):

        start_time = time.time()

        sCTEM.sample_b_ij(nodepair, eventtime)
        sCTEM.sample_u_ij(nodepair, eventtime)

        sCTEM.back_propagate_pis_betas(nodepair, eventtime, pis0)

        sCTEM.sample_C_i(nodepair, eventtime)

        sCTEM.sample_Lambda_k1k2(nodepair, eventtime)

        sCTEM.sample_alpha(eventtime)

        sCTEM.sample_delta(nodepair, eventtime, ite)
        running_time.append(time.time()-start_time)
        # sCTEM.ll_cal(nodepair, eventtime)

        # print('running time is: ', time.time()-start_time)
        # print('log likelihood is: ', sCTEM.ll_val)

        # ll_seq.append(sCTEM.ll_val)

        if (ite >burnInTime):

            mean_Lambdas = (mean_Lambdas*(ite - burnInTime-1)+sCTEM.Lambdas)/(ite-burnInTime)
            mean_betas = (mean_betas*(ite - burnInTime-1)+sCTEM.betas)/(ite-burnInTime)
            for ci in range(len(sCTEM.pis_list)):
                for di in range(len(sCTEM.pis_list[ci])):
                    mean_pi[ci][di] = (mean_pi[ci][di] * (ite - burnInTime - 1) + sCTEM.pis_list[ci][di]) / (ite - burnInTime)
                    mean_C[ci][di] = (mean_C[ci][di] * (ite - burnInTime - 1) + sCTEM.C_i_list[ci][di]) / (ite - burnInTime)
            seq_delta.append(sCTEM.Delta_pis)
            seq_tau.append(sCTEM.Taus_kij)
            seq_alpha.append(sCTEM.alpha)

            if np.mod(ite, 2)==0:
                np.savez_compressed(pathss_save+name+str(ids)+'.npz', running_time = running_time, mean_Lambdas = mean_Lambdas, mean_betas = mean_betas, \
                                    mean_pi = mean_pi, mean_C = mean_C, seq_alpha = seq_alpha, seq_tau = seq_tau, seq_delta = seq_delta, \
                                    IterationTime = IterationTime, NumIteration = NumEdge, dataNum = dataNum, KK = KK)
