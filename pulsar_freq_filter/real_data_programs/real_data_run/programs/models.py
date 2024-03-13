import numpy as np
from kalman import KalmanFilterTimeVarying, KalmanFilterTimeVarying1

def param_map(params):
    A, B, C, D = params['ratio'], (params['tau']), params['lag'], params['omgc_dot']
    taus = (1+A) * B
    tauc = (1+A) / A * B
    Qc = params['Qc']
    Qs = params['Qs']
    Nc = (D + A/(1+A) * C/B)
    Ns = D - C/B * (1+A)**-1
    return tauc, taus, Qc, Qs, Nc, Ns

def param_map2(params):
    A, B, C, D = params['ratio'], (params['tau']), params['lag'], -params['neg_omgc_dot']
    taus = (1+A) * B
    tauc = (1+A) / A * B
    Qc = params['Qc']
    Qs = params['Qs']
    Nc = (D + A/(1+A) * C/B)
    Ns = D - C/B * (1+A)**-1
    return tauc, taus, Qc, Qs, Nc, Ns

def construct_Q_fast(tauc, taus, Qc, Qs, dts):
    tau = 1/(1/tauc + 1/taus)
    expvals = np.exp(-dts/tau)
    Q = np.zeros((2, 2, dts.size))
    Q[0,0,:] = (Qc * tauc**2 + Qs * taus**2) * dts +\
               (2 * Qc *tauc * taus - 2 * Qs * taus**2) * tau * (1 - expvals) +\
               (Qc * taus**2 + Qs * taus**2) * (1 - expvals**2) * (tau / 2)
    Q[1,1,:] = (Qc * tauc**2 + Qs * taus**2) * dts +\
               (2 * Qs *tauc * taus - 2 * Qc * tauc**2) * tau * (1 - expvals) +\
               (Qc * tauc**2 + Qs * tauc**2) * (1 - expvals**2) * (tau / 2)
    Q[1,0,:] = (Qs * taus**2 + Qc * tauc**2) * dts + \
               (Qc * tauc * taus - Qc * tauc**2 + Qs * tauc * taus - Qs * taus**2) * tau * (1 - expvals) - \
               (Qc + Qs) * (tauc * taus) * (1 - expvals**2) * (tau/2)
    Q[0,1,:]  = Q[1, 0, :].copy()
    return Q / (tauc + taus)**2

def construct_F_fast(tauc, taus, dts):
    tau = 1/(1/tauc + 1/taus)
    expvals = np.exp(-dts / tau)
    F = np.zeros((2,2,dts.size))
    F[0, 0, :] = tauc + taus * expvals
    F[0, 1, :] = taus - taus * expvals
    F[1, 0, :] = tauc - tauc * expvals
    F[1, 1, :] = taus + tauc * expvals
    return F / (tauc + taus)

def construct_B_fast(tauc, taus, Nc, Ns, dts):
    tau = (tauc * taus) / (tauc + taus)
    expvals = np.exp(-dts / tau)
    omg_dot = (tauc * Nc + taus * Ns) / (tauc + taus)
    B = np.zeros((2, dts.size))
    B[0, :] = omg_dot * dts + \
              tau**2 * (tauc**-1 * (Nc - Ns) * (1 - expvals))
    B[1, :] = omg_dot * dts + \
              tau**2 * (taus**-1 * (Ns - Nc) * (1 - expvals))
    return B

class TwoComponentModel(KalmanFilterTimeVarying):
    #R = measurement error covariance list, C = design/emission/observation matrix list
    def __init__(self, times, data, R=None, C=None, param_map_function=None, solve=True):
        super(TwoComponentModel, self).__init__()
        self.times = times
        self.data = data
        self.R = R.copy()
        self.C = C
        self.param_map = param_map_function
        self.nmeasurements, self.nstates = np.shape(self.C)
        self.solve = solve
        self.R_original = self.R.copy()
    def update_parameters(self, params):
        tauc, taus, Qc, Qs, Nc, Ns = self.param_map(params)
        dts = self.times[1:] - self.times[:-1]
        self.F = construct_F_fast(tauc, taus, dts)
        self.Q = construct_Q_fast(tauc, taus, Qc, Qs, dts)
        self.B  = construct_B_fast(tauc, taus, Nc, Ns, dts)
        self.R[0, 0, :] = self.R_original[0, 0, :] * params['EFAC'] + params['EQUAD']
    def loglike(self, params, loglikelihood_burn=-1, return_states=False):
        self.update_parameters(params)
        tauc, taus, Qc, Qs, Nc, Ns = self.param_map(params)
        lag = (tauc * taus / (tauc + taus)) * (Nc - Ns)
        if params["omgs_0"] == None:
            omgc_0 = params["omgc_0"]
            omgs_0 = omgc_0 - lag
        else:
            omgc_0 = params["omgc_0"]
            omgs_0 = params["omgs_0"]
        return self.ll_on_data(self.data[1:], params, x0=np.array([omgc_0, omgs_0]),
                               P0=np.eye(self.nstates)*np.max(self.R[:, :, 0]),
                               loglikelihood_burn=loglikelihood_burn, return_states=return_states)



class OneComponentModel(KalmanFilterTimeVarying1):
    def __init__(self, times, data, R=None):
        super(OneComponentModel, self).__init__()
        self.times = times
        self.data = data
        self.R = R
        self.nmeasurements = 1
        self.nstates = 1
        self.R_original = self.R.copy()
    def update_parameters(self, params):
        Qc = params['Qc']
        omgc_dot = params['omgc_dot']
        dts = self.times[1:] - self.times[:-1]
        self.F = np.ones(len(dts))
        self.Q = Qc*dts
        self.B = omgc_dot*dts
        self.R = self.R_original * params['EFAC'] + params['EQUAD']
    def loglike(self, params, loglikelihood_burn=-1, return_states=False):
        self.update_parameters(params)
        omgc_0 = params["omgc_0"]
        return self.ll_on_data(self.data[1:], params, x0=omgc_0, P0=self.R[0],
                               loglikelihood_burn=loglikelihood_burn,
                               return_states=return_states)

