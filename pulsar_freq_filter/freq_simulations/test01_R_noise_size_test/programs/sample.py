import numpy as np
import bilby

class KalmanLikelihood(bilby.Likelihood):
    def __init__(self, neutron_star_model, parameters=None):
        if parameters is None:
            parameters={'ratio': None, 'tau': None, 'Qc': None, 'Qs': None, 
                        'lag': None, 'omgc_dot': None, 'omgc_0': None, 
                        'omgs_0': None, 'EFAC': None, 'EQUAD': None}
        super().__init__(parameters=parameters)
        self.neutron_star_model = neutron_star_model
    
    def log_likelihood(self):
        try:
            ll = self.neutron_star_model.loglike(self.parameters, loglikelihood_burn=-1)
        except np.linalg.LinAlgError:
            ll= -np.inf
        if np.isnan(ll):
            ll = -np.inf
        return ll
