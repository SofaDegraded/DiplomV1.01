import numpy as np
"""
    Класс для считывания:
    - начальных параметров,
    - приращений деградационного показателя,
    - оценок параметров.
"""
class ReadInitData:

    '''
     Считывание параметров из файла.

     flag = 0 - считывание при наличии начальных параметров
     для модели с "fix"-эффектом
     flag = 1 - считывание при наличии начальных параметров
     для модели с "rnd"-эффектом
     flag = 2 - считывание времен замера деградац. показателя
     gamma1 = .0 # параметр формы для IG
     gamma2 = .0 # показатель степени функции тренда
     sigma = .0 # параметр масштаба для IG
     ksi = .0 # параметр формы для Gamma
     eta = .0 # параметр масштаба для Gamma
     N = 0 # количество замеров
     K = 0 # объем выборки
     time = [] #время замера дегр. показателя
    '''
    def reading_init_param(self, fileName, flag):
        str = np.loadtxt(fileName)
        if flag == 0:
            gamma1, gamma2, sigma, N, K = str[0], str[1], str[2], str[3], str[4]
            time = str[5:]
            return gamma1, gamma2, sigma, N, K, time
        elif flag == 1:
            gamma1, gamma2, xi, eta = str[0], str[1], str[2], str[3]
            N, K = str[4], str[5]
            time = str[6:]
            return gamma1, gamma2, xi, eta, N, K, time
        else:
            N, K = str[0], str[1]
            time = str[2:]
            return N, K, time
    '''
     Считывание приращений деградационного показателя из файла
     либо показателей деградации.

     flag = 0 - считывание для смоделированной выборки приращений
     flag = 1 - считывание показателей деградации для лазеров
    '''
    def reading_incr_or_degrad_ind(self, fileName, flag, M=None, N=None, K=None):
        if flag == 0:
            # 1 - M, 2 - N, 3 - K
            dZ = np.loadtxt(fileName).reshape((M, N - 1, K))
        else:
            dZ = np.loadtxt(fileName)
        return dZ
    '''
     Считывание оценок параметров.

     model_type = 0 - считывание для модели с "fix"-эффектом
     model_type = 1 - считывание для модели с "rnd"-эффектом
    '''
    def reading_est(self, fileName, model_type):    
        str = np.loadtxt(fileName)
        if model_type == 0:
            est_gamma1, est_gamma2, est_sigma = str[:, 0], str[:, 1], str[:, 2]
            return est_gamma1, est_gamma2, est_sigma
        else:
            est_gamma1, est_gamma2, est_ksi, est_eta = str[:, 0], str[:, 1], str[:, 2], str[:, 3]
            return est_gamma1, est_gamma2, est_ksi, est_eta
   
"""
    Класс для записи результатов в файл.
""" 
class WriteData:
    '''
    Запись данных в файл.
    В случае, если подается 3D массив, необходимо привести в 2D.
    Только для записи данных с одинаковой длиной.
    '''
    @staticmethod
    def writing_in_file_with_same_length(fileName, data):
        np.savetxt(fileName, data, fmt='%.8e', delimiter=' ', newline='\n')

    '''
    Запись данных в файл.
    Для записи данных с различной длиной.
    '''
    @staticmethod
    def writing_in_file_with_var_length(fileName, data1, data2):
        with open(fileName,"w") as f:
            f.write("\n".join(" ".join(map(str, x)) for x in (data1, data2)))

