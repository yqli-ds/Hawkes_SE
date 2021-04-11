import numpy as np
import scipy
import copy
import scipy.io as scio
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from scipy.stats import poisson, norm, gamma, dirichlet, uniform, beta, invgamma
import math
from scipy.special import logsumexp



def initialize_model(NN, KK, nodepair, eventtime):
    # Input:
    # KK: number of communities
    # NN: number of identities （N）
    # data: data file len(event)*3 (ii to jj at time tt)

    # Output:
    # pis: N X KK: initial mixed-membership distributions
    # pis_list:
    # C_i_list: list containing NN of list each has J_i X KK: layer-wise latent counting statistics matrix

    # betas: N X N: information propagation coefficient with parameters a_betas, b_betas
    # Lambdas: KK X KK: community compatibility matrix with parameters a_Lambda, b_Lambda
    # M with parameters a_M, b_M
    # hyper parameters: Delta_pis, Taus_kij, alpha with parameters(a_alpha, b_alpha)
    # b_ij

    Delta_pis = 1
    Taus_kij = 1

    #### M
    a_M = 1000
    b_M = 1
    M = gamma.rvs(a=a_M, scale=b_M)


    #### alpha in Eq 1 to calculate the tt
    a_alpha = 100
    b_alpha = 0.01
    alpha = gamma.rvs(a=a_alpha, scale=b_alpha)

    #### Lambdas K * K
    a_Lambda = 1
    b_Lambda = 1 / (M * NN)
    Lambdas = gamma.rvs(a = a_Lambda, scale = 1, size = (KK, KK))/(M**2)

    #### betas N * N
    a_betas = 1
    b_betas = 1
    betas = gamma.rvs(a=a_betas, scale=b_betas, size=(NN, NN))


    pis0 = np.ones((NN, KK))*0.1

    pis = dirichlet.rvs(alpha=np.ones(KK)*0.1, size=NN)

    #### pis_list，N list, pis_list[ii][0] = pis[ii], 后面的项目会在接下来的程序中update
    pis_list = [[pis[ii]] for ii in range(NN)]



    #### C_i_0，N list, C_i_list[ii][0] ~ poisson(pis[ii](M * pis_list[ii][0])), 后面的项目会在接下来的程序中update
    C_i_list=[[(poisson.rvs(M * pis[ii])+1).astype(int)] for ii in range(NN)]


    ########################################################################################################
    ########################################################################################################

    ######
    # length = len(eventtime), for each [tt] [sum ii, and sum jj], calculating index, eg, pis_list[ii](+1,(+ current ii)),pis[jj]
    sender_receiver_num = []
    # length = len(eventtime), for each [tt], before tt where(data index) sender == jj and receiver == ii
    mutually_exciting_pair = []


    receiving_j_list = []
    receiving_j_list_0 = []
    receiving_j_list_1 = []
    receiving_j_list_2 = []


    time_inteval_receive_j_list = []

    time_inteval_non_receive_j_list = []

    time_inteval_receive_j_to_i_list = []


    sending_time_list = [eventtime[nodepair[:, 0]==ii] for ii in range(NN)]


    #### b_ij counts for each Eij： (N,)
    b_ij = np.zeros(len(eventtime)).astype(int)

    for tt in range(len(eventtime)):
        # current_data = data[:(tt)] # not including the data[tt]
        current_pair = nodepair[:tt]
        # current_time = eventtime[:tt]

        #### saving receiving_j_list, update pis_list[1:], C_i_list[1:]
        # unique_jj send to ii before [tt]
        unique_jj = np.unique(current_pair[(current_pair[:, 1] == nodepair[tt, 0]), 0])
        i_parameter_contribute_from_J = np.zeros(KK)
        if len(unique_jj) > 0:
            receiving_i = []
            receiving_i_0 = []
            receiving_i_1 = []
            receiving_i_2 = []
            for nn in range(len(unique_jj)):
                j_current = (unique_jj[nn])
                # before tt, data index of j_current as sender
                j_location_in_data = np.where(current_pair[:, 0] == j_current)[0]
                # j_index = before[tt], index of j_current(to i) in j list，取最后一个, 计算pis或C取哪一个
                j_pi_position = (np.where(current_pair[j_location_in_data, 1]==nodepair[tt, 0])[0])[-1]
                # saving current receiving_j_list[tt] from j_current
                receiving_i.append([j_current, j_pi_position, eventtime[j_location_in_data[j_pi_position]]])
                receiving_i_0.append(j_current)
                receiving_i_1.append(j_pi_position)
                receiving_i_2.append(eventtime[j_location_in_data[j_pi_position]])
                # for updating  pis_list[ii] contribution from jj
                i_parameter_contribute_from_jj = betas[j_current, (nodepair[tt, 0])] * np.exp(
                    -Delta_pis * (eventtime[tt]-receiving_i[nn][2])) * (pis_list[j_current][receiving_i[nn][1]])
                i_parameter_contribute_from_J += i_parameter_contribute_from_jj
            #### updating receiving_j_list
            receiving_j_list.append(receiving_i)
            receiving_j_list_0.append(receiving_i_0)
            receiving_j_list_1.append(receiving_i_1)
            receiving_j_list_2.append(receiving_i_2)
        else:
            receiving_j_list.append([])
            receiving_j_list_0.append([])
            receiving_j_list_1.append([])
            receiving_j_list_2.append([])

        #### updating pis_list[ii]
        i_parameter_contribute_from_prei = betas[(nodepair[tt, 0]), (nodepair[tt, 0])] * (pis_list[(nodepair[tt, 0])][-1])

        para_nn = i_parameter_contribute_from_J + i_parameter_contribute_from_prei + 1e-6

        pis_i = gamma.rvs(a=para_nn, scale=1) + 1e-16
        pis_i = (pis_i / np.sum(pis_i))

        pis_list[(nodepair[tt, 0])].append(pis_i)
        #### updating C_i_list[(nodepair[tt, 0])]
        C_i_list[(nodepair[tt, 0])].append((poisson.rvs(M * pis_list[(nodepair[tt, 0])][-1]) +1).astype(int))

        #### saving sender_receiver_num
        sender_num = np.sum(current_pair[:, 0] == nodepair[tt, 0]) # before[tt] sum ii
        receiver_num = np.sum(current_pair[:, 0] == nodepair[tt, 1]) # before[tt] sum jj
        sender_receiver_num.append([sender_num, receiver_num])
        # before[tt] sum ii
        # sender_receiver_num.append(sender_num)
        # updating mutually_exciting_pair(用于update b_ij)
        mutually_index = np.where((current_pair[:, 0] == nodepair[tt, 1])&(current_pair[:, 1] == nodepair[tt, 0]))[0]
        # if len(mutually_index)>1:
        #     a = 1
        mutually_exciting_pair.append(mutually_index) # data locations, before tt, where jj to ii in data

        if len(mutually_exciting_pair[tt])>0:
            base_intensity = (C_i_list[(nodepair[tt, 0])][sender_receiver_num[tt][0]][np.newaxis, :].dot(Lambdas).dot \
                                             (C_i_list[(nodepair[tt, 1])][sender_receiver_num[tt][1]][:, np.newaxis]))[0]
            # base_intensity = pis_list[(nodepair[tt, 0])][-1].dot(Lambdas).dot(pis_list[(nodepair[tt, 1])][-1])
            excitation_function_val = alpha*np.exp(-Taus_kij*(eventtime[tt]-eventtime[mutually_exciting_pair[tt]]))
            prob = np.concatenate((base_intensity, excitation_function_val))
            # b_ij[tt] = np.random.choice((len(mutually_exciting_pair[tt])+1),1,p=prob/np.sum(prob))
            b_ij[tt] = np.random.choice((len(mutually_exciting_pair[tt])+1), p=prob / np.sum(prob))


        if sender_num == 0:
            s_time = 0
            # time index between current(-) and 0(+)
            tt_index = np.where((eventtime >= s_time) & (eventtime < eventtime[tt]))[0]
        else:

            s_time = sending_time_list[(nodepair[tt, 0])][sender_num-1]

            tt_index = np.where((eventtime > s_time) & (eventtime < eventtime[tt]))[0]
        # data between current(-) and previous(-)/0（+）
        covered_pair = nodepair[tt_index, :]

        #### saving time_inteval_receive_j_list

        unique_jj = np.unique(covered_pair[:, 0])
        jj_time_interval_tt = []
        jj_to_i_time_interval_tt = []
        for jj in unique_jj:
            # data index of jj_sender in between current(-) and previous(-)/0（+）
            tt_index_jj = tt_index[covered_pair[:, 0]==jj].tolist()
            # 记录此范围内j_sender和他在data中的位置
            jj_time_interval_tt.append([jj, tt_index_jj])
            #### saving j_sender_i_receiver_list:
            if np.sum(nodepair[tt_index_jj,1]==nodepair[tt,0])>0:
                index_j_to_i = (np.asarray(tt_index_jj))[nodepair[tt_index_jj,1]==nodepair[tt,0]].tolist()
                jj_to_i_time_interval_tt.append([jj, index_j_to_i])
        time_inteval_receive_j_list.append(jj_time_interval_tt)
        time_inteval_receive_j_to_i_list.append(jj_to_i_time_interval_tt)
        #### saving time_inteval_non_receive_j_list （node list remove tt above and ii)

        non_receive_tt = [ii for ii in range(NN) if ii not in unique_jj]

        non_receive_tt.remove((nodepair[tt, 0]))

        time_inteval_non_receive_j_list.append(non_receive_tt)

    # for updating C_i_list: all node last time point to T
    last_time_inteval_non_receive_j_list = []
    last_time_inteval_receive_j_list = []
    last_time_inteval_receive_j_to_i_list = []
    for ii in range(NN):
        if (len(sending_time_list[ii]))==0:
            s_time = 0
            # time index between current(-) and 0(+)
            last_index = np.where((eventtime >= s_time) & (eventtime <= eventtime[-1]))[0]
        else:
            s_time = sending_time_list[ii][-1]
            # time index between current(-) and 0(+)
            last_index = np.where((eventtime > s_time) & (eventtime <= eventtime[-1]))[0]

        covered_pair = nodepair[last_index, :]
        #### saving time_inteval_receive_j_list

        unique_jj = np.unique(covered_pair[:, 0])
        jj_time_interval_last = []
        jj_to_i_time_interval_last = []
        for jj in unique_jj:
            # data index of jj_sender in between current(-) and previous(-)/0（+）
            last_index_jj = last_index[covered_pair[:, 0] == jj].tolist()

            jj_time_interval_last.append([jj, last_index_jj])

            if np.sum(nodepair[last_index_jj, 1] == ii) > 0:
                index_j_to_i = (np.asarray(last_index_jj))[nodepair[last_index_jj, 1] == ii].tolist()
                jj_to_i_time_interval_last.append([jj, index_j_to_i])
        last_time_inteval_receive_j_list.append(jj_time_interval_last)
        last_time_inteval_receive_j_to_i_list.append(jj_to_i_time_interval_last)

        non_receive_last = [ii for ii in range(NN) if ii not in unique_jj]

        non_receive_last.remove(ii)
        last_time_inteval_non_receive_j_list.append(non_receive_last)





    ########################################################################################################
    ########################################################################################################


    return time_inteval_receive_j_list, time_inteval_non_receive_j_list,sending_time_list, sender_receiver_num, \
            mutually_exciting_pair, receiving_j_list, time_inteval_receive_j_to_i_list, sending_time_list,  pis_list, \
           pis0, C_i_list, b_ij, betas, a_betas, b_betas, Lambdas, a_Lambda, b_Lambda, M, a_M, b_M, alpha, a_alpha, \
           b_alpha, Delta_pis, Taus_kij, last_time_inteval_non_receive_j_list, last_time_inteval_receive_j_list, last_time_inteval_receive_j_to_i_list


class sCTEM_class:
    def __init__(self, NN,  KK,  pis_list, C_i_list, time_inteval_receive_j_list, time_inteval_non_receive_j_list, \
                 b_ij, betas, a_betas, b_betas, Lambdas, a_Lambda, b_Lambda, M,a_M, b_M, alpha, a_alpha, b_alpha, Delta_pis, \
                 Taus_kij, sending_time_list, sender_receiver_num,mutually_exciting_pair, receiving_j_list, \
                 time_inteval_receive_j_to_i_list, last_time_inteval_non_receive_j_list, \
                 last_time_inteval_receive_j_list, last_time_inteval_receive_j_to_i_list):
        self.NN = NN
        self.KK = KK
        self.b_ij = b_ij

        self.M = M
        self.a_M = a_M
        self.b_M = b_M
        self.betas = copy.copy(betas)
        self.a_betas = a_betas
        self.b_betas = b_betas
        self.Lambdas = copy.copy(Lambdas)
        self.a_Lambda = a_Lambda
        self.b_Lambda = b_Lambda

        self.alpha = alpha
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha

        self.C_i_list = copy.copy(C_i_list)
        self.pis_list = copy.copy(pis_list)

        self.Delta_pis = Delta_pis
        self.Taus_kij = Taus_kij

        self.sending_time_list = sending_time_list
        self.sender_receiver_num = sender_receiver_num
        self.mutually_exciting_pair = mutually_exciting_pair
        self.receiving_j_list = receiving_j_list

        self.time_inteval_receive_j_to_i_list = time_inteval_receive_j_to_i_list
        self.time_inteval_receive_j_list = time_inteval_receive_j_list
        self.time_inteval_non_receive_j_list = time_inteval_non_receive_j_list

        self.last_time_inteval_non_receive_j_list = last_time_inteval_non_receive_j_list
        self.last_time_inteval_receive_j_list = last_time_inteval_receive_j_list
        self.last_time_inteval_receive_j_to_i_list = last_time_inteval_receive_j_to_i_list

    def back_propagate_pis_betas(self, nodepair, eventtime, pis0):
        m_ik_list = copy.deepcopy(self.C_i_list)

        z_ik_j_sum_k = np.zeros((self.NN, self.NN))
        beta_b_hyper = np.zeros((self.NN, self.NN))

        for tt in range(len(eventtime)-1, -1, -1):

            psi_kk_list = np.zeros(((len(self.receiving_j_list[tt])+1), self.KK))

            if len(self.receiving_j_list[tt]) > 0:

                exp_time_current_s = np.zeros(len(self.receiving_j_list[tt]))

                for nn in range(len(self.receiving_j_list[tt])):
                    exp_time_current_s[nn] = np.exp(-self.Delta_pis * (eventtime[tt]-self.receiving_j_list[tt][nn][2]))
                    psi_kk_list[nn] = self.betas[self.receiving_j_list[tt][nn][0], (nodepair[tt, 0])] * exp_time_current_s[nn] * (
                                                         self.pis_list[self.receiving_j_list[tt][nn][0]][self.receiving_j_list[tt][nn][1]])



            psi_kk_list[-1] = self.betas[(nodepair[tt,0]), (nodepair[tt,0])] * (self.pis_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]])

            psi_sum = np.sum(psi_kk_list, axis=0)

            para_nn = (psi_sum + m_ik_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]+1]) + 1e-10

            pis_i_current = gamma.rvs(a=para_nn, scale=1) + 1e-16
            self.pis_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]+1] = (pis_i_current / np.sum(pis_i_current))


            latent_count_i = np.sum(m_ik_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]+1]).astype(float) + 1e-10

            psi_para_1 = np.sum(psi_sum)+ 1e-10

            gam1 = gamma.rvs(a = [psi_para_1, latent_count_i], scale = 1)+1e-10
            q_i_s = gam1[0]/np.sum(gam1)

            if len(self.receiving_j_list[tt]) > 0:
                for nn in range(len(self.receiving_j_list[tt])):
                    beta_b_hyper[self.receiving_j_list[tt][nn][0],(nodepair[tt, 0])] += (exp_time_current_s[nn] * np.log(q_i_s))
            beta_b_hyper[nodepair[tt,0],(nodepair[tt, 0])] += (np.log(q_i_s))

            for kk in range(self.KK):
                current_val = m_ik_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]+1][kk]
                if current_val> 0:

                   y_i_k = np.sum(uniform.rvs(size=current_val) < (psi_sum[kk] / (psi_sum[kk] + np.arange(current_val))))
                   z_ik_J = np.random.multinomial(y_i_k, psi_kk_list[:, kk] / psi_sum[kk])

                   if len(self.receiving_j_list[tt]) > 0:
                      for nn in range(len(self.receiving_j_list[tt])):
                          m_ik_list[self.receiving_j_list[tt][nn][0]][self.receiving_j_list[tt][nn][1]][kk] += z_ik_J[nn]

                          z_ik_j_sum_k[self.receiving_j_list[tt][nn][0], (nodepair[tt,0])] += z_ik_J[nn]

                   m_ik_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]][kk] += z_ik_J[-1]
                   z_ik_j_sum_k[(nodepair[tt,0]), (nodepair[tt,0])] += z_ik_J[-1]


        m_ii = np.asarray([m_ik_list[ii][0] for ii in range(self.NN)])
        para_nn = (pis0 + m_ii).astype(float)
        nn_pis = gamma.rvs(a=para_nn, scale=1) + 1e-16
        for ii in range(self.NN):
            self.pis_list[ii][0] = (nn_pis[ii] / np.sum(nn_pis[ii]))

        self.betas = gamma.rvs(a=(self.a_betas + z_ik_j_sum_k).astype(float), scale=1) / (self.b_betas - beta_b_hyper)


    def cal_jj_integral(self,jj, jj_time, jj_num, s_time, splus1_time):
        # if len(jj_time)==0:
        #     integral_val = self.C_i_list[jj][jj_num]*(splus1_time-s_time)
        if len(jj_time)==1:
            integral_val_1 = self.C_i_list[jj][jj_num[0]]*(jj_time[0]-s_time)
            integral_val_2 = self.C_i_list[jj][jj_num[0]+1]*(splus1_time-jj_time[0])
            integral_val = integral_val_1 + integral_val_2
        elif len(jj_time)>1:
            integral_val_1 = self.C_i_list[jj][jj_num[0]]*(jj_time[0]-s_time)
            integral_val_2 = self.C_i_list[jj][jj_num[-1]+1]*(splus1_time-jj_time[-1])
            integral_val_3 = np.zeros(self.KK)
            for ij in range(len(jj_time)-1):
                integral_val_3 += self.C_i_list[jj][jj_num[ij]+1]*(jj_time[ij+1]-jj_time[ij])
            integral_val = integral_val_1 + integral_val_2 + integral_val_3
        return integral_val

    def sample_u_ij(self, nodepair, eventtime):
        # uij_tt：len(event) * 2
        self.u_ij = np.zeros((len(eventtime), 2)).astype(int)
        for tt in range(len(eventtime)):
            pois_uij = ((self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]])[:, np.newaxis])*\
                       ((self.C_i_list[(nodepair[tt, 1])][self.sender_receiver_num[tt][1]])[np.newaxis, :])*self.Lambdas
            # normalize pois_uij
            pois_uij = pois_uij / np.sum(pois_uij)
            # u_ij  categorical(0-(kk*kk-1))
            try:
                categorical = np.random.choice(a=len(pois_uij.reshape(-1)), p=pois_uij.reshape(-1))
            except:
                a = 1

            # change categorical to the location in a kk*kk matrix, 这个值赋给u_ij[tt]
            row = int(categorical / self.KK)
            column = int(categorical - self.KK * row)
            self.u_ij[tt][0] = row
            self.u_ij[tt][1] = column



    def sample_C_i(self, nodepair, eventtime):

        current_C_val = np.asarray([self.C_i_list[ii][0] for ii in range(self.NN)])
        # elapse_time_1 = 0
        # elapse_time_2 = 0
        for tt in range(len(eventtime)):

            splus1_time = eventtime[tt]

            if self.sender_receiver_num[tt][0] == 0:

                s_time = 0
            else:

                s_time = self.sending_time_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]-1]


            integral_val_tt = np.zeros(self.KK)

            for jj_val in self.time_inteval_receive_j_list[tt]:
                jj = jj_val[0]
                jj_time = eventtime[jj_val[1]]  # a list of index in data
                # jj_num = [self.sender_receiver_num[jj_val[1][nn]][0] for nn in range(len(jj_val[1]))]
                jj_num = [self.sender_receiver_num[jj_val_1_n][0] for jj_val_1_n in jj_val[1]]
                integral_val_tt += self.cal_jj_integral(jj, jj_time, jj_num, s_time, splus1_time)


            integral_val_tt += np.sum(current_C_val[self.time_inteval_non_receive_j_list[tt]], axis=0)*(splus1_time - s_time)

            ll_previous_num = np.zeros(self.KK)
            if len(self.time_inteval_receive_j_to_i_list[tt]) > 0:
                for jj_val in self.time_inteval_receive_j_to_i_list[tt]:
                    jj_val_num = np.sum((self.b_ij[jj_val[1]] == 0)[:, np.newaxis] * (
                            (self.u_ij[jj_val[1]][:, 1][:, np.newaxis]) == (np.arange(self.KK)[np.newaxis, :])),
                                        axis=0)
                    ll_previous_num += jj_val_num


            ll_current_num = (self.b_ij[tt] == 0) * (self.u_ij[tt][0] == np.arange(self.KK))
            for k in range(self.KK):
                prior = self.M * (self.pis_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]][k])
                a1 = -np.sum(integral_val_tt*(self.Lambdas[:, k]+self.Lambdas[k, :]))
                a2 = ll_previous_num[k] + ll_current_num[k]

                self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]][k] = self.touchard_sample(a1, a2, prior)
            current_C_val[nodepair[tt,0]] = self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]+1]


        for ii in range(self.NN):
            if ii!=((nodepair[-1,0])):
                splus1_time = eventtime[-1]
                ## ii never send to other j
                if (len(self.sending_time_list[ii]))==0:
                    s_time = 0
                else:

                    s_time = self.sending_time_list[ii][-1]


                integral_val_last = np.zeros(self.KK)

                for jj_val in self.last_time_inteval_receive_j_list[ii]:
                    jj = jj_val[0]
                    jj_time = eventtime[jj_val[1]]  # a list of index in data
                    jj_num = [self.sender_receiver_num[jj_val_1_n][0] for jj_val_1_n in jj_val[1]]
                    integral_val = self.cal_jj_integral(jj, jj_time, jj_num, s_time, splus1_time)
                    integral_val_last += integral_val


                integral_val_last += np.sum(current_C_val[self.last_time_inteval_non_receive_j_list[ii]], axis=0)*(splus1_time-s_time)
                # if (np.sum(abs(vv-vv_2)))>1e-9:
                #     print('Wrong')


                ll_previous_num = np.zeros(self.KK)
                if len(self.last_time_inteval_receive_j_to_i_list[ii]) > 0:
                    for jj_val in self.last_time_inteval_receive_j_to_i_list[ii]:
                        jj_val_num = np.sum((self.b_ij[jj_val[1]] == 0)[:, np.newaxis] * (
                                (self.u_ij[jj_val[1]][:, 1][:, np.newaxis]) == (np.arange(self.KK)[np.newaxis, :])),
                                            axis=0)
                        ll_previous_num += jj_val_num


                ll_current_num = np.zeros(self.KK) #(self.b_ij[tt] == 0) * (self.u_ij[tt][0] == np.arange(self.KK))

                for k in range(self.KK):
                    prior = self.M * (self.pis_list[ii][-1][k])
                    a1 = -np.sum(integral_val_last * (self.Lambdas[:, k]+self.Lambdas[k]))   # likelihood 第1项
                    a2 = ll_previous_num[k] + ll_current_num[k]  # likelihood 第二，三项指数的和
                    self.C_i_list[ii][-1][k] = self.touchard_sample(a1, a2, prior)
                current_C_val[ii] = self.C_i_list[ii][-1]

        # print('running time 3 is: ', time.time() - start_time)

    def touchard_sample(self, a1, a2, prior):
        if a2 == 0:
            select_val = poisson.rvs(prior*np.exp(a1))
        else:
            candidates = np.arange(1, 1000 + 1)  # self.NN, we did not consider 0 because the ratio is 0 for sure
            pseudos = candidates * (a1+np.log(prior)) + a2 * np.log(candidates) - np.cumsum(np.log(candidates))
            proportions = np.exp(pseudos - logsumexp(pseudos))
            select_val = np.random.choice(candidates, p=proportions )
        return select_val


    def sample_Lambda_k1k2(self, nodepair, eventtime):
        # sample Lambda according to the gamma distribution

        # gamma distribution b_Lambda update, based on equation(19)
        area = np.zeros((self.KK, self.KK))


        C_ii = np.asarray([self.C_i_list[ii][0] for ii in range(self.NN)])


        for tt in range(len(eventtime)-1):

            splus1_time_range = eventtime[tt+1]-eventtime[tt]

            C_ii[(nodepair[tt, 0])] = self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0] + 1]
            first_val = np.sum(C_ii, axis=0)[:, np.newaxis] * np.sum(C_ii, axis=0)[np.newaxis, :] - C_ii.T.dot(C_ii)
            area += first_val * splus1_time_range

        self.area = area

        # gamma distribution a_Lambda update, based on equation(19)
        judge_sum = np.zeros((self.KK, self.KK))
        for kk1 in range(self.KK):
            for kk2 in range(self.KK):
                # 整个event中bij==0 * uij==kk1,kk2
                judge_sum[kk1, kk2] = np.sum((self.b_ij == 0).reshape((-1))*((self.u_ij[:,0].astype(int)==kk1).reshape((-1)) * (self.u_ij[:,1].astype(int)==kk2).reshape((-1))))

        self.Lambdas = gamma.rvs(a = self.a_Lambda + judge_sum, scale = 1)/(self.b_Lambda + area)


    def sample_alpha(self, eventtime):
        # Updating the hyper-parameter alpha

        # calculate h_tau
        h_tau = np.sum((self.Taus_kij**(-1))*(1-np.exp(-self.Taus_kij*(eventtime[-1]-eventtime))))

        # sum all bij >0
        b_ij_over_1_sum = np.sum(self.b_ij > 0)

        # equation(15)
        self.alpha = gamma.rvs(a = self.a_alpha + b_ij_over_1_sum, scale = 1)/(self.b_alpha + h_tau)

    def sample_b_ij(self, nodepair, eventtime):
        # sampling the latent integers b_ij counts for each Eij： (N,)

        for tt in range(len(eventtime)):
            if len(self.mutually_exciting_pair[tt]) > 0: # 存在 jj to ii pair
                base_intensity = (self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]]).dot(self.Lambdas).dot(
                    (self.C_i_list[(nodepair[tt, 1])][self.sender_receiver_num[tt][1]]))
                excitation_function_val = self.alpha * np.exp(
                    -self.Taus_kij * (eventtime[tt] - eventtime[self.mutually_exciting_pair[tt]]))
                prob = np.concatenate(([base_intensity], excitation_function_val))
                self.b_ij[tt] = np.random.choice((len(self.mutually_exciting_pair[tt]) + 1), p=(prob / np.sum(prob)))
        self.b_ij = self.b_ij.astype(int)

    def sample_delta(self, nodepair, eventtime, ite):

        a_delta = 0.1
        b_delta = 0.1
        a_taus = 0.1
        b_taus = 0.1
        delta_old = self.Delta_pis
        delta_new = delta_old + norm.rvs()*((np.sqrt(ite+1))**(-1))

        taus_old = self.Taus_kij
        taus_new = taus_old + norm.rvs()*((np.sqrt(ite+1))**(-1))

        if delta_new>0:

            ll_old = 0
            ll_new = 0

            for tt in range(len(eventtime)):
                i_parameter_contribute_from_J_old = np.zeros(self.KK)
                i_parameter_contribute_from_J_new = np.zeros(self.KK)
                if len(self.receiving_j_list[tt]) > 0:

                    for nn in range(len(self.receiving_j_list[tt])):
                        exp_time_current_s_old = np.exp(-delta_old * (eventtime[tt]-self.receiving_j_list[tt][nn][2]))
                        val_1_old = self.betas[self.receiving_j_list[tt][nn][0], (nodepair[tt, 0])] * exp_time_current_s_old * (
                                                             self.pis_list[self.receiving_j_list[tt][nn][0]][self.receiving_j_list[tt][nn][1]])
                        i_parameter_contribute_from_J_old += val_1_old

                        exp_time_current_s_new = np.exp(-delta_new * (eventtime[tt]-self.receiving_j_list[tt][nn][2]))
                        val_1_new = self.betas[self.receiving_j_list[tt][nn][0], (nodepair[tt, 0])] * exp_time_current_s_new * (
                                                             self.pis_list[self.receiving_j_list[tt][nn][0]][self.receiving_j_list[tt][nn][1]])
                        i_parameter_contribute_from_J_new += val_1_new


                i_parameter_contribute_from_prei = self.betas[(nodepair[tt,0]), (nodepair[tt,0])] * (self.pis_list[(nodepair[tt,0])][self.sender_receiver_num[tt][0]])
                psi_i_s_old = i_parameter_contribute_from_J_old + i_parameter_contribute_from_prei
                psi_i_s_new = i_parameter_contribute_from_J_new + i_parameter_contribute_from_prei

                ll_old +=dirichlet.logpdf(self.pis_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]+1], psi_i_s_old)
                ll_new +=dirichlet.logpdf(self.pis_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]+1], psi_i_s_new)

            ll_old += gamma.logpdf(delta_old, a = a_delta, scale = b_delta)
            ll_new += gamma.logpdf(delta_new, a = a_delta, scale = b_delta)

            if np.log(np.random.rand())<(ll_new-ll_old):
                self.Delta_pis = delta_new

        if taus_new>0:
            ll_old = 0
            ll_new = 0

            judge = np.where(self.b_ij>0)[0]
            b_nonzero = self.b_ij[judge]
            receiving_nozero1 = [self.mutually_exciting_pair[judge_i] for judge_i in judge]

            receiving_time = [eventtime[receiving_nozero1[it][b_nonzero[it]-1]] for it in range(len(b_nonzero))]

            ll_old += np.sum(-taus_old*(eventtime[judge]-receiving_time))
            ll_new += np.sum(-taus_new*(eventtime[judge]-receiving_time))

            ll_old -= self.alpha*np.sum((taus_old**(-1)) * (1 - np.exp(-taus_old* (eventtime[-1] - eventtime))))
            ll_new -= self.alpha*np.sum((taus_new**(-1)) * (1 - np.exp(-taus_new* (eventtime[-1] - eventtime))))

            ll_old += gamma.logpdf(taus_old, a=a_taus, scale=b_taus)
            ll_new += gamma.logpdf(taus_new, a=a_taus, scale=b_taus)

            if np.log(np.random.rand()) < (ll_new - ll_old):
                self.Taus_kij = taus_new

    def ll_cal(self, nodepair, eventtime):
        ll_val = 0

        judge = np.where(self.b_ij > 0)[0]
        b_nonzero = self.b_ij[judge]
        receiving_nozero1 = [self.mutually_exciting_pair[judge_i] for judge_i in judge]
        receiving_time = [eventtime[receiving_nozero1[it][b_nonzero[it] - 1]] for it in range(len(b_nonzero))]

        ll_val -= np.sum(self.Taus_kij* (eventtime[judge] - receiving_time))
        ll_val += len(receiving_time)*np.log(self.alpha)

        judge = np.where(self.b_ij == 0)[0]

        for tt in judge:
            pois_uij_ss = ((self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0]]))[:, np.newaxis]*\
                       ((self.C_i_list[(nodepair[tt, 1])][self.sender_receiver_num[tt][1]]))[np.newaxis, :]*self.Lambdas
            ll_val += np.log(pois_uij_ss[self.u_ij[tt][0]][self.u_ij[tt][1]])

        area = np.zeros((self.KK, self.KK))

        C_ii = np.asarray([self.C_i_list[ii][0] for ii in range(self.NN)])

        for tt in range(len(eventtime)-1):

            splus1_time_range = eventtime[tt+1]-eventtime[tt]

            C_ii[(nodepair[tt, 0])] = self.C_i_list[(nodepair[tt, 0])][self.sender_receiver_num[tt][0] + 1]
            first_val = np.sum(C_ii, axis=0)[:, np.newaxis] * np.sum(C_ii, axis=0)[np.newaxis, :] - C_ii.T.dot(C_ii)
            area += first_val * splus1_time_range

        ll_val -= np.sum(area*self.Lambdas)

        h_tau = np.sum((self.Taus_kij**(-1))*(1-np.exp(-self.Taus_kij*(eventtime[-1]-eventtime))))
        ll_val -= self.alpha*h_tau

        self.ll_val = ll_val