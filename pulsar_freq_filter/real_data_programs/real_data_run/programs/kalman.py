import numpy as np
from numpy.linalg import inv, det, solve, slogdet
from numba import jit

@jit(nopython=True)
def predict(x, P, F, B, Q):
    xp = F.dot(x) + B
    Pp = F @ P @ F.T + Q
    return xp, Pp

@jit(nopython=True)
def update_solve(xp, Pp, y, C, R, nmeasurements, nstates):
    err = y - C @ xp
    S = C @ Pp @ C.T + R
    K = solve(S.T, (Pp @ C.T).T).T
    x = xp + K @ err
    P = (np.eye(nstates) - K @ C) @ (Pp)
    ll = -0.5 * (slogdet(S)[1] + err.T @ solve(S, err) + nmeasurements * np.log(2*np.pi))
    return x, P, ll

class KalmanFilterTimeVarying(object):
    def __init__(self):
        pass

    def predict(self, timestep):
        self.xp, self.Pp = predict(self.x.reshape((self.nstates, 1)), self.P, 
                                   self.F[:, :, timestep], 
                                   self.B[:, timestep].reshape((self.nstates, 1)),
                                   self.Q[:, :, timestep])

    def update(self, y, timestep):
        self.x, self.P, ll = update_solve(self.xp.reshape((self.nstates, 1)), self.Pp, 
                                          y.reshape((self.nmeasurements, 1)), 
                                          self.C, self.R[:, :, timestep+1], 
                                          self.nmeasurements, self.nstates)
        self.ll += ll.squeeze()

    def ll_on_data(self, data, params=None, x0=None, P0=None, loglikelihood_burn=-1, return_states=False):
        self.ll = 0
        self.x = x0
        self.P = P0
        Nobs, Ndim = np.shape(data)
        if return_states:
            xs = np.zeros((Nobs, self.nstates))
            Ps = np.zeros((Nobs, self.nstates, self.nstates))
            xps = np.zeros((Nobs, self.nstates))
            Pps = np.zeros((Nobs, self.nstates, self.nstates))
            lls = np.zeros(Nobs)
        for nn in range(0, Nobs):
            if nn <= loglikelihood_burn:
                self.ll = 0
            self.predict(nn)
            self.update(data[nn, :], nn)
            if return_states:
                xs[nn, :] = self.x.reshape(self.nstates)
                Ps[nn, :, :] = self.P.reshape((self.nstates, self.nstates))
                xps[nn, :] = self.xp.reshape(self.nstates)
                Pps[nn, :, :] = self.Pp.reshape((self.nstates,self.nstates))
                lls[nn] = self.ll
        #Return the results
        if return_states:
            return self.ll, xs, Ps, xps, Pps, lls
        else:
            return self.ll

    def update_parameters(self, params):
        pass



class KalmanFilterTimeVarying1(object):
    def __init__(self):
        pass

    def predict(self, timestep):
        x = self.x
        P = self.P
        F = self.F[timestep]
        B = self.B[timestep]
        Q = self.Q[timestep]

        xp = F*x + B
        Pp = F*P*F + Q

        self.xp = xp
        self.Pp = Pp

    def update(self, measurement, timestep):
        xp = self.xp
        Pp = self.Pp
        y = measurement
        R = self.R[timestep]

        err = y - xp
        x = xp + (Pp/(Pp+R)) * err
        P = (R/(Pp+R)) * Pp
        ll = -0.5 * (np.log(2*np.pi*(Pp+R)) + err**2/(Pp+R))

        self.x = x
        self.P = P
        self.ll_change = ll
        self.ll += ll

    def ll_on_data(self, data, params=None, x0=None, P0=None, loglikelihood_burn=-1, return_states=False):
        self.ll = 0
        self.x = x0
        self.P = P0
        Nobs = data.size
        if return_states:
            xs = np.zeros(Nobs)
            Ps = np.zeros(Nobs)
            xps = np.zeros(Nobs)
            Pps = np.zeros(Nobs)
            lls = np.zeros(Nobs)
        for i in range(0, Nobs):
            if i <= loglikelihood_burn:
                self.ll = 0
            self.predict(i)
            self.update(data[i], i)
            if return_states:
                xs[i] = self.x
                Ps[i] = self.P
                xps[i] = self.xp
                Pps[i] = self.Pp
                lls[i] = self.ll_change
        if return_states:
            return self.ll, xs, Ps, xps, Pps, lls
        else:
            return self.ll

