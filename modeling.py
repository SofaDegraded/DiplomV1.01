import numpy as np
import random as rand
import scipy as sp
import scipy.stats as spstat

import reading as r

"""
    Класс для работы с обратным гауссовским распределением 
    при наличии параметров формы и масштаба.
"""

class InvgaussGen(spstat.rv_continuous):
    """
    Генерация случайных чисел согласно обратному гауссовскому распределению.
    """
    def _rvs(self, mu, sigma):
        return self._random_state.wald(mu, sigma, size=self._size)

    """
    Плотность распределения вероятностей для обратного гауссовского распределения.
    """
    def _pdf(self, x, mu, sigma):
        return sqrt(sigma / (2*pi*x**3.0))*\
        exp(- sigma / (2*x) * ((x - mu) / mu)**2.0)

    """
    Функция распределения вероятностей для обратного гауссовского распределения.
    """
    def _cdf(self, x, mu, sigma):
        fac = sqrt(sigma / x)
        C1 = _norm_cdf(fac * (x - mu) / mu)
        C1 += exp(2*sigma / mu) * _norm_cdf(- fac * (x + mu) / mu)
        return C1

"""
Класс для моделирования выборок приращений деградационного показателя.
"""
class ModelingDegradationData:
    '''
    Конструктор класса.

    InitData - экземпляр класса, отвечающего за считывание параметров
    flag = 0 - получение параметров для модели с "fix"-эффектом
    flag = 1 - получение параметров для модели с "rnd"-эффектом
    flag = 2 - получение времен замера деградац. показателя
    '''
    def __init__(self, InitData, flag): 
        if flag == 0:
             gamma1, gamma2, sigma, N, K, time = InitData.reading_init_param('input.txt', flag)
             self.gamma1, self.gamma2, self.sigma = gamma1, gamma2, sigma
             self.N, self.K, self.time = int(N), int(K), time
        elif flag == 1:
            gamma1, gamma2, xi, eta, N, K, time = InitData.reading_init_param('input.txt', flag)
            self.gamma1, self.gamma2, self.xi, self.eta = gamma1, gamma2, xi, eta
            self.N, self.K, self.time = int(N), int(K), time
        else:
            N, K, time = InitData.reading_init_param('input.txt', flag)
            self.N, self.K, self.time = int(N), int(K), time

    '''
    Cтепенная функция тренда.

    time - время замера дегр. показателя
    '''
    def func_trend(self, time): 
        return time ** self.gamma2

    '''
    Генерация случайного параметра сигма,
    распределенного по Gamma-распределению с параметрами кси и эта
    для IG-модели с "rnd"-эффектом.
    '''    
    def rnd_volatility_model(self):
        rnd_sigma = np.array([rand.gammavariate(self.xi, self.eta) for i in range(self.N - 1)])
        return rnd_sigma

    '''
    Вычисление параметров для IG-модели.

    model_type = 0 - получение параметра b для модели с "fix"-эффектом
    model_type = 1 - получение параметра b для модели с "rnd"-эффектом
    '''   
    def degr_model_param(self, model_type): 
        a = list(map(lambda x, y: self.gamma1 * (self.func_trend(x) - \
                 self.func_trend(y)), self.time[1:], self.time[:-1]))
        if model_type == 0:
            b = list(map(lambda x, y: self.sigma * (self.func_trend(x) - \
                     self.func_trend(y)) ** 2, self.time[1:], self.time[:-1]))
        elif model_type == 1:
            rnd_sigma = self.rnd_volatility_model()
            b = list(map(lambda x, y, z: z * (self.func_trend(x) - \
                     self.func_trend(y)) ** 2, self.time[1:], self.time[:-1], rnd_sigma))
        return np.array(a), np.array(b)

    '''
    Вычисление приращений деградационных показателей 
    для IG-модели для K объектов и N замеров.

    model_type = 0 - получение параметров для модели с "fix"-эффектом
    model_type = 1 - получение параметров для модели с "rnd"-эффектом
    '''
    def modeling_deltaZ(self, model_type): 
        a, b = self.degr_model_param(model_type)
        inversegauss = InvgaussGen(name='invgauss')
        deltaZ = np.array([inversegauss.rvs(a[i], b[i], size=self.K) for i in range(self.N - 1)])
        return deltaZ

    '''
    Вычисление показателей деградации.
    Z(0) = 0

    deltaZ - приращения деград. показателя
    '''
    def rate_of_degrad(self, deltaZ):  
        rate_degr = [[0]*self.K]
        rate_degr.extend(list(zip(*[sp.cumsum(deltaZ[:, i]) for i in range(self.K)])))
        return np.array(rate_degr)

    '''
    Вычисление приращений при наличии показателей деградации.

    Z - выборка деград. показателей
    '''
    def calc_incr_of_degrad_ind(self, Z):          
        deltaZ = np.diff(Z, axis=0)
        return deltaZ