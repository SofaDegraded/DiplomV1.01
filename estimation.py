import math as mth
import numpy as np
from scipy.optimize import minimize
import time as t

from mpi4py import MPI

import reading as r

'''
Класс для вычисления оценок максимального правдоподобия.
'''
class EstimationOfParameters:
    '''
    Конструктор класса.

    ModDegData - экземпляр класса, отвечающего за моделирование выборки
    model_type = 0 - получение параметров для модели с "fix"-эффектом
    model_type = 1 - получение параметров для модели с "rnd"-эффектом
    model_type = 2 - получение времен замера деградац. показателя
    '''
    def __init__(self, ModDegData, model_type): 
        if model_type == 0:
             self.gamma1, self.gamma2, self.sigma = ModDegData.gamma1, \
                                                    ModDegData.gamma2, ModDegData.sigma
             self.N, self.K, self.time = int(ModDegData.N), int(ModDegData.K), ModDegData.time
        elif model_type == 1:
            self.gamma1, self.gamma2 = ModDegData.gamma1, ModDegData.gamma2
            self.xi, self.eta = ModDegData.xi, ModDegData.eta
            self.N, self.K, self.time = int(ModDegData.N), int(ModDegData.K), ModDegData.time
        else:
            self.N, self.K, self.time = int(ModDegData.N), int(ModDegData.K), ModDegData.time

    '''
    Логарифмическая функция правдоподобия для модели с "fix"-эффектом.

    est_theta - оценки параметров (значение симплекса)
    deltaZ - выборка приращений деград. показателя
    amountpar = 0 - оценивается модель со степенной функцией тренда (3 параметра)
    amountpar = 1 - оценивается модель со линейной функцией тренда (2 параметра)
    '''
    def likelihood_function_fix_eff(self, est_theta, deltaZ, amountpar):
        if amountpar == 0:
            temp = np.array(list(map(lambda x, y: x ** est_theta[1] - \
                            y ** est_theta[1], self.time[1:], self.time[:-1])))
            tmp = np.array(list(map(lambda x, y, z: (z - est_theta[0] * \
                          (x ** est_theta[1] - y ** est_theta[1])) ** 2, \
                           self.time[1:], self.time[:-1], deltaZ)))
            dT = np.log(temp)
            ds = np.sum(tmp / deltaZ)
            dzlog = np.sum(np.log(deltaZ))
            lnL = (self.N - 1) * (self.K) * np.log(est_theta[2]) / 2 + \
                  (self.K) * np.sum(dT) - (self.N - 1) * (self.K) * np.log(2 * mth.pi) / 2 - \
                  3 * dzlog / 2 - (est_theta[2] / (2 * est_theta[0] ** 2)) * ds
        else:
            temp = np.array(list(map(lambda x, y: x  - y , self.time[1:], self.time[:-1])))
            tmp = np.array(list(map(lambda x, y, z: (z - est_theta[0] * (x  - y)) ** 2, \
                           self.time[1:], self.time[:-1], deltaZ)))
            dT = np.log(temp)
            ds = np.sum(tmp / deltaZ)
            dzlog = np.sum(np.log(deltaZ))
            lnL =  (self.N - 1) * (self.K) * np.log(est_theta[1] if est_theta[1]>0 else 1) / 2 + \
                   (self.K) * np.sum(dT) - (self.N - 1) * (self.K) * np.log(2 * mth.pi) / 2 - \
                    3 * dzlog / 2 - (est_theta[1] / (2 * est_theta[0] ** 2)) * ds
        return -lnL

    '''
    Логарифмическая функция правдоподобия для модели с "rnd"-эффектом.

    est_theta - оценки параметров (значение симплекса)
    deltaZ - выборка приращений деград. показателя
    amountpar = 0 - оценивается модель со степенной функцией тренда (4 параметра)
    amountpar = 1 - оценивается модель со линейной функцией тренда (3 параметра)
    '''
    def likelihood_function_rnd_eff(self, est_theta, deltaZ, amountpar):
        if amountpar == 0:
            tmp = np.array(list(map(lambda x, y, z: est_theta[3] * (z - est_theta[0] * \
                          (x ** est_theta[1] - y ** est_theta[1])) ** 2 + \
                          (2 * (est_theta[0]** 2) * z ), self.time[1:], self.time[:-1], deltaZ)))

            temp = np.array(list(map(lambda x, y: x ** est_theta[1] - \
                            y ** est_theta[1], self.time[1:], self.time[:-1])))
            try:
                dT = np.sum(np.log(temp))
                dslog1 = (self.N - 1) * self.K * (mth.log(2) + \
                          2 * mth.log(est_theta[0] if est_theta[0]>0 else 1) + \
                          mth.log(est_theta[3] if est_theta[3]>0 else 1)) + np.sum(np.log(deltaZ))
                dslog = np.sum(np.log(tmp))
                lnL = (self.N - 1) * (self.K) * mth.log(mth.gamma(est_theta[2] + 1/2)) +\
                      (est_theta[2] + 0.5) * dslog1 + self.K * dT - \
                      (self.N - 1) * (self.K) * mth.log(mth.gamma(est_theta[2])) - \
                      (self.N - 1) * (self.K)  * mth.log(2 * mth.pi) / 2 - \
                       3 * np.sum(np.log(deltaZ)) / 2 - \
                      (est_theta[2] + 0.5) * dslog -\
                      (self.N - 1) * (self.K) * est_theta[2] * mth.log(est_theta[3] if est_theta[3]>0 else 1) 
            
            except:
                print('EXCEPTION')
                lnL = 0
            else:
                return -lnL
        else:
            tmp = np.array(list(map(lambda x, y, z: est_theta[2] * \
                          (z - est_theta[0] * (x  - y)) ** 2 + \
                          (2 * (est_theta[0]** 2) * z ), self.time[1:], self.time[:-1], deltaZ)))

            temp = np.array(list(map(lambda x, y: x  - y , self.time[1:], self.time[:-1])))
            try:
                dT = np.sum(np.log(temp))
                dslog1 = (self.N - 1) * self.K * (mth.log(2) + \
                          2 * mth.log(est_theta[0] if est_theta[0]>0 else 1) + \
                          mth.log(est_theta[2] if est_theta[2]>0 else 1)) + np.sum(np.log(deltaZ))
                dslog = np.sum(np.log(tmp))
                lnL = (self.N - 1) * (self.K) * mth.log(mth.gamma(est_theta[1] + 1/2)) +\
                      (est_theta[1] + 0.5) * dslog1 + self.K * dT - \
                      (self.N - 1) * (self.K) * mth.log(mth.gamma(est_theta[1])) - \
                      (self.N - 1) * (self.K)  * mth.log(2 * mth.pi) / 2 - \
                       3 * np.sum(np.log(deltaZ)) / 2 - \
                      (est_theta[1] + 0.5) * dslog -\
                      (self.N - 1) * (self.K) * est_theta[1] * mth.log(est_theta[2] if est_theta[2]>0 else 1) 
            
            except:
                print('EXCEPTION!!!')
                lnL = 0
            else:
                return -lnL
        return -lnL


    '''
    Нахождение экстремума логарифмической функции правдоподобия
    с помощью метода безусловной оптимизации - Нелдера-Мида.

    deltaZ - выборка приращений дегра. показателя
    model_type = 0 - вычисление оценок параметров для модели с "fix"-эффектом
    model_type = 1 - вычисление оценок параметров для модели с "rnd"-эффектом
    amountpar = 0 - оценивается модель со степенной функцией тренда (3 параметра)
    amountpar = 1 - оценивается модель со линейной функцией тренда (2 параметра)
    est_theta0 - начальное приближение
    '''
    def max_likelihood_estimation(self, deltaZ, model_type, amountpar):
        if model_type == 0:   
            if amountpar == 0:    
                est_theta0 = np.array([0.0019, 1., 5.e-05])
            else: 
                est_theta0 = np.array([0.0019, 5.e-05])
            lam_fun = lambda est_theta: self.likelihood_function_fix_eff(est_theta, deltaZ, amountpar)
            result = minimize(lam_fun, est_theta0, method='nelder-mead', \
                              options={'xtol': 1e-12, 'disp': True})
           
        else:
            if amountpar == 0: 
                est_theta0 = np.array([.002, 1., 170., 3.e-07])
            else:
                est_theta0 = np.array([.002, 171., 3.e-07])
            lam_fun = lambda est_theta: self.likelihood_function_rnd_eff(est_theta, deltaZ, amountpar)
            result = minimize(lam_fun, est_theta0, method='nelder-mead', \
                              options={'xtol': 1e-12, 'maxfev':400, 'disp': True})
        return [result.x, result.fun]

    '''
    Исследование свойств ОМП параметров IG-модели
    с помощью имитационного метода Монте-Карло.
    Для анализа точности ОМП вычисляется величина psi.

    ModDegData - экземпляр класса, отвечающего за моделирование выборки
    M - количество повторений выборки приращений и вычисления ОМП
    model_type = 0 - вычисление оценок параметров для модели с "fix"-эффектом
    model_type = 1 - вычисление оценок параметров для модели с "rnd"-эффектом
    amountpar = 0 - оценивается модель со степенной функцией тренда (3 параметра)
    amountpar = 1 - оценивается модель со линейной функцией тренда (2 параметра)
    '''
    def method_Monte_Karlo(self, InitData, ModDegData, M, model_type, amountpar, flag):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if flag == 0:
            tmp_deltaZ = InitData.reading_incr_or_degrad_ind('dZ.txt', flag, M, self.N, self.K)  

        if size != 0:
            proc_M = int(M / size) # количество повторений выборки для каждого процесса
            if flag == 0:
                deltaZ = tmp_deltaZ[rank * proc_M : (rank + 1) * proc_M]
            else:
                # моделирование выборки приращений  
                deltaZ = np.array([ModDegData.modeling_deltaZ(model_type) for i in range(proc_M)])
            # оценивание параметров модели
            est_maxlnL = [self.max_likelihood_estimation(x, model_type, amountpar) for x in deltaZ]
            est_param_maxlnL = np.array([est_maxlnL[i][0] for i in range(proc_M)]) # ОМП
            lnL = np.array([est_maxlnL[i][1] for i in range(proc_M)]) # логарифм. функция правдоподобия

            # вычисление величины psi
            if model_type == 0:
                g1, sigm = est_param_maxlnL[:, 0], est_param_maxlnL[:, -1]
                if amountpar == 0:
                    g2 = est_param_maxlnL[:, 1]
                    psi_gamma2 = np.sum(abs(g2 - self.gamma2) / self.gamma2) / proc_M

                psi_gamma1 = np.sum(abs(g1 - self.gamma1) / self.gamma1) / proc_M
                psi_sigma = np.sum(abs(sigm - self.sigma) / self.sigma) / proc_M
                
            else:
                g1,  xi_eta = est_param_maxlnL[:, 0],  \
                              est_param_maxlnL[:, -2] * est_param_maxlnL[:, -1]
                if amountpar == 0:
                    g2 = est_param_maxlnL[:, 1]
                    psi_gamma2 = np.sum(abs(g2 - self.gamma2) / self.gamma2) / proc_M

                psi_gamma1 = np.sum(abs(g1 - self.gamma1) / self.gamma1) / proc_M
                psi_sigma = np.sum(abs(xi_eta - self.xi * self.eta) / (self.xi * self.eta)) / proc_M

            #сбор величины psi со всех процессов
            if amountpar == 0:
                psi_est_par = np.array([psi_gamma1, psi_gamma2, psi_sigma])
                psi_est = np.empty(3, dtype=np.double)
            else: 
                psi_est_par = np.array([psi_gamma1, psi_sigma])
                psi_est = np.empty(2, dtype=np.double)
            comm.Allreduce(psi_est_par, psi_est, op=MPI.SUM) 

            #сбор приращений со всех процессов
            dZ = np.zeros((M, self.N - 1, self.K))
            comm.Gather(deltaZ, dZ, root=0) #сбор результатов со всех процессов

            #сбор ОМП со всех процессов
            if model_type == 0:
                if amountpar == 0:
                    est = np.zeros((M, 3))
                else: 
                    est = np.zeros((M, 2))
            else:
                if amountpar == 0:
                    est = np.zeros((M, 4))
                else:
                    est = np.zeros((M, 3))
            comm.Gather(est_param_maxlnL, est, root=0) 

            #сбор логарифма функции правдоподобия со всех процессов
            new_lnL = np.zeros(M)
            comm.Gather(lnL, new_lnL, root=0)

            if rank == 0:
                r.WriteData().writing_in_file_with_same_length('dZ.txt', dZ.reshape(-1, dZ.shape[-1]))
                r.WriteData().writing_in_file_with_same_length('est_par.txt', est)
                r.WriteData().writing_in_file_with_same_length('psi.txt', psi_est / size)
                r.WriteData().writing_in_file_with_same_length('lnL.txt', new_lnL)
        MPI.Finalize()
        return psi_est / size
   
    '''
    Оценивание ОМП параметров при наличии данных о деградации
    на одной выборке.

    InitData - экземпляр класса, отвечающего за считывание параметров
    ModDegData - экземпляр класса, отвечающего за моделирование выборки
    flag = 1 - считывание показателей деградации лазеров
    '''
    def estimate_dataZ(self, InitData,  ModDegData, flag=1):
        amountpar = 1 #т.к. линейная функция тренда
        #считывание показателей деградации
        Z = InitData.reading_incr_or_degrad_ind('deltaZz.txt', flag) 
        #преобразование в приращения дегр. показателя
        deltaZ = ModDegData.calc_incr_of_degrad_ind(Z)  
        #оценивание параметров как для модели с "fix" & "rnd" эффектами
        est_param_maxlnL_fix, lnL_fix = np.array(self.max_likelihood_estimation(deltaZ, 0, amountpar))
        est_param_maxlnL_rnd, lnL_rnd = np.array(self.max_likelihood_estimation(deltaZ, 1, amountpar))
        #запись в файл
        r.WriteData().writing_in_file_with_var_length('est_par_fix.txt', est_param_maxlnL_fix, np.array([lnL_fix]))
        r.WriteData().writing_in_file_with_var_length('est_par_rnd.txt', est_param_maxlnL_rnd, np.array([lnL_rnd]))
        return est_param_maxlnL_fix, lnL_fix, est_param_maxlnL_rnd, lnL_rnd