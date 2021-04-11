import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# namess = ['ege','mpo', 'erf', 'ubun']
namess = ['ege']

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

    total_auc = []
    total_precision = []
    name_seq = []
    ids = 0
    for file_i in filess:
        name_i = file_i[13:19]
        name_seq.append(name_i)

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
            for ii in range(NN):
                historical_time_pair.append([])
                for jj in range(NN):
                    historical_time_pair[ii].append([])

            for tt in range(data.shape[0]):
                historical_time_pair[int(data[tt, 0])][int(data[tt, 1])].append(data[tt, 2])
            repeat_time = 100
            repeat_precision = []
            repeat_auc = []
            for rr in range(repeat_time):
                rr_interval = np.sort(np.random.uniform(low = eventtime_test[0], high = eventtime_test[-1], size = 2))
                rr_index = (eventtime_test>=rr_interval[0])&(eventtime_test<rr_interval[1])
                nodepair_rr = nodepair_test[rr_index]

                rate_val = np.zeros((NN, NN))
                true_val = np.zeros((NN, NN))
                true_val[nodepair_rr[:, 0], nodepair_rr[:, 1]] = 1
                for ii in range(NN):
                    for jj in range(NN):
                        if ii!=jj:
                            base_rate = (mean_C[ii][-1].dot(mean_Lambdas).dot(mean_C[jj][-1]))*(rr_interval[1]-rr_interval[0])
                            if len(historical_time_pair[jj][ii])>0:
                                cu_jj_ii = np.asarray(historical_time_pair[jj][ii])
                                base_rate += np.sum(seq_alpha[-1]/seq_tau[-1]*(np.exp(-seq_tau[-1]*(rr_interval[0]-cu_jj_ii))-np.exp(-seq_tau[-1]*(rr_interval[1]-cu_jj_ii))))
                            rate_val[ii][jj] = base_rate

                rate = np.ravel(rate_val)
                ground = np.ravel(true_val)
                ground = ground[rate!=0]
                rate = rate[rate!=0]

                repeat_precision.append(average_precision_score(ground, rate))
                repeat_auc.append(roc_auc_score(ground, rate))
                print('Dataset: ', ni)
                print('repeat finish: ', rr)
                print('precision: ', repeat_precision[-1])
                print('auc: ', repeat_auc[-1])
                print('+++++++++++++')
                print(' ')
            total_auc.append(repeat_auc)
            total_precision.append(repeat_precision)

            ids = np.random.randn()
            np.savez_compressed(''+ni+str(ids)+'.npz', total_precision = total_precision, total_auc=total_auc)
