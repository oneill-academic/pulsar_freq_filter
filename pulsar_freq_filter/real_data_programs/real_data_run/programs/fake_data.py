import sdeint
import numpy as np

def two_component_fake_data(times, tauc, taus, sigmac, sigmas, 
                            Nc, Ns, omgc_0, omgs_0, Rc, Rs):
    tau = (tauc * taus) / (tauc + taus)
    F_int = np.array([[-1/tauc, 1/tauc],[1/taus, -1/taus]])
    def f(x, t):
        return F_int.dot(x) + np.array([Nc, Ns])
    def g(x, t):
        return np.diag([sigmac, sigmas])
    states = sdeint.itoint(f, g, np.array([omgc_0, omgs_0]), times)
    data = states.copy()
    data[:, 0] += np.random.randn(times.size) * np.sqrt(Rc)
    data[:, 1] += np.random.randn(times.size) * np.sqrt(Rs)
    return data, states

def one_component_fake_data(times, sigmac, omgc_dot, omgc_0, Rc):
    def f(x, t):
        return omgc_dot
    def g(x, t):
        return sigmac
    states = sdeint.itoint(f, g, omgc_0, times)
    data = states.copy()[:, 0]
    data += np.random.randn(times.size) * np.sqrt(Rc)
    return data, states

