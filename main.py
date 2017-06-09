import estimation as e
import modeling as m
import reading as r


if __name__ == '__main__':
    M = 10
    model_type = 1

    init_data = r.ReadInitData()
    model = m.ModelingDegradationData(init_data, model_type)
    est = e.EstimationOfParameters(model, model_type)
    est.method_Monte_Karlo(init_data, model, M, 1, 1, 0)
