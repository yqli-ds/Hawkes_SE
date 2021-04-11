import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

namess = ['mpo', 'erf', 'ubun', 'ege']

for ni in namess:
    filess = glob.glob(''+ni+'*.npz')

    save_path = ''
    pathss = ''

    filess_data = glob.glob(pathss +'*'+ ni+'*.txt')

    filess_data_0 = filess_data[0]
    data_ori = (pd.read_csv(filess_data_0, sep=" ", header=None).values).astype(float)

    data_test = data_ori[10001:30000]
    data = data_ori[:10000]
    data = data[(data[:, 0] != data[:, 1])]
    unique_id = np.unique(data[:, :2])  # np.sort(np.unique(data[:, :2]))
    data_id = data[:, :2]
    for ii in range(len(unique_id)):
        data_id[data_id == unique_id[ii]] = ii
    data[:, 2] = (data[:, 2] / (24 * 3600.0))
    min_time = np.min(data[:, 2])
    data[:, 2] = data[:, 2] - np.min(data[:, 2])
    NN = len(unique_id)
    sending_time_list = [data[data[:, 0] == ii, 2] for ii in range(NN)]

    lenss = np.asarray([len(send_i) for send_i in sending_time_list])
    pseudo_len = np.sort(lenss)
    max_id = np.where(lenss == pseudo_len[-1])[0][0]
    max_id_2 = np.where(lenss == pseudo_len[-2])[0][0]
    time_union = np.sort(np.unique(np.concatenate((sending_time_list[max_id_2], sending_time_list[max_id]))))

    total_ll = []

    ids = 0
    for file_i in filess:

        result = np.load(file_i, allow_pickle=True)
        IterationTime = result['IterationTime']
        mean_Lambdas = result['mean_Lambdas']
        running_time = result['running_time']
        mean_betas = result['mean_betas']
        mean_pi = result['mean_pi']
        mean_C = result['mean_C']
        seq_alpha = result['seq_alpha']
        seq_tau = result['seq_tau']
        seq_delta = result['seq_delta']
        NumEdge = result['NumIteration']
        dataNum = result['dataNum']

        if IterationTime == 1000:
            nodepair_test = data_test[:, :2].astype(int)
            eventtime_test = data_test[:, 2]

            eventtime_test = eventtime_test[nodepair_test[:, 0]!=nodepair_test[:, 1]]
            nodepair_test = nodepair_test[nodepair_test[:, 0]!=nodepair_test[:, 1]]

            replace = np.ones(nodepair_test.shape)
            for ii in range(len(unique_id)):
                replace[nodepair_test==unique_id[ii]] = 0
                nodepair_test[nodepair_test == unique_id[ii]] = ii
            nodepair_test = nodepair_test[np.sum(replace, axis=1)==0]
            eventtime_test = eventtime_test[np.sum(replace, axis=1)==0]
            eventtime_test = eventtime_test/((24 * 3600.0))
            eventtime_test -= min_time

            historical_time_pair = []
            test_matrix = []
            for ii in range(NN):
                historical_time_pair.append([])
                test_matrix.append([])
                for jj in range(NN):
                    historical_time_pair[ii].append([])
                    test_matrix[ii].append([])

            for tt in range(data.shape[0]):
                historical_time_pair[int(data[tt, 0])][int(data[tt, 1])].append(data[tt, 2])


            for tt in range(len(nodepair_test)):
                test_matrix[nodepair_test[tt, 0]][nodepair_test[tt, 1]].append(eventtime_test[tt])

            ll = 0
            for ii in range(NN):
                for jj in range(NN):
                    if ii!=jj:
                        base_rate = (mean_C[ii][-1].dot(mean_Lambdas).dot(mean_C[jj][-1]))
                        ll -= base_rate*(eventtime_test[-1]-eventtime_test[0])
                        if len(historical_time_pair[jj][ii])>0:
                            cu_jj_ii = np.asarray(historical_time_pair[jj][ii])
                            ll -= np.sum(seq_alpha[-1]/seq_tau[-1]*(np.exp(-seq_tau[-1]*(eventtime_test[0]-cu_jj_ii))-np.exp(-seq_tau[-1]*(eventtime_test[-1]-cu_jj_ii))))

                        if len(test_matrix[ii][jj])>0:
                            for test_ij_time in test_matrix[ii][jj]:
                                if len(historical_time_pair[jj][ii]) > 0:
                                    cu_jj_ii = np.asarray(historical_time_pair[jj][ii])
                                    ij_rate = base_rate + np.sum(seq_alpha[-1] * (np.exp(-seq_tau[-1] * (test_ij_time - cu_jj_ii))))
                                else:
                                    ij_rate = base_rate
                                ll += np.log(ij_rate)

            print('Dataset: ', ni)
            print('DataNum: ', dataNum)
            print('T: ', data[-1, 2])
            print('ll: ', ll)
            print('+++++++++++++')
            print(' ')
            total_ll.append(ll)

    ids = np.random.randn()
    np.savez_compressed('ll_'+ni+str(ids)+'.npz', total_ll = total_ll)
