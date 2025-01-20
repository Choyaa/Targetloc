import numpy as np
from lib.sample import get_sampler
def get_protocol(sampler_name, scaler_name, protocol_conf =None, scaler_conf = None, multi_seed = False):
    sampler, scaler = get_sampler(sampler_name, scaler_name, scaler_conf=scaler_conf)
    if protocol_conf is None:
        protocol_conf = {
                        'N_steps': 20,
                        'n_views': 20,
                        'M': 6 
                        }
    protocol = 'Single_seed'
    if not multi_seed:
        protocol_class = Protocol1
        
    else :
        protocol_class = Protocol2
    protocol_obj = protocol_class(protocol_conf, sampler, scaler, protocol)
    return protocol_obj

    
class BaseProtocol:
    """This base dummy class serves as template for subclasses. it always returns
    the same poses without perturbing them"""
    def __init__(self, conf, sampler, scaler, protocol_name):
        self.sampler = sampler
        self.scaler = scaler
        self.n_steps = conf['N_steps']
        self.n_views = conf['n_views']
        self.protocol = protocol_name
        # init for later
        self.center_std = None
        self.max_angle = None
        
    def init_step(self, i):
        self.scaler.step(i)
        self.center_std, self.max_angle = self.scaler.get_noise()
    
    def get_pertubr_str(self, step, res):
        c_str = "_".join(list(map(lambda x: f'{x:.1f}'.replace('.', ','), map(float, self.center_std))))
        angle_str = "_".join(list(map(lambda x: f'{x:.1f}'.replace('.', ','), map(float, self.max_angle))))

        perturb_str = f'pt{self.protocol}_s{step}_sz{res}_theta{angle_str}_t{c_str}'
        return perturb_str
    
    @staticmethod
    def get_r_name(q_name, ref, beam_i):
        r_name =f'{q_name}_{ref}_{beam_i}'
        return r_name
    
    def resample(self):
        pass
        


class Protocol1(BaseProtocol):
    """
    This protocol keeps only the first prediction, to perturb N times
    """
    def __init__(self, conf, scaler, sampler, protocol_name):
        super().__init__(conf, scaler, sampler, protocol_name)
    
    # override
    def resample(self, q_name, pred_t, pred_e, seed_e, seed_t, beam_i=0):
                                           
        render_es = []
        render_ts = []
        r_names = []
        
        old_t = pred_t  # take first prediction
        old_e = pred_e  # take first prediction

        #### keep previous estimate #####
        render_es.append(old_e)
        render_ts.append(old_t)
        r_names.append(BaseProtocol.get_r_name(q_name, 0, beam_i))
        ####################
        views_per_candidate = self.n_views - 1

        new_es, new_ts = self.sampler.sample_batch(views_per_candidate, 
                                            self.center_std, 
                                            self.max_angle, 
                                            old_t, 
                                            old_e,
                                            seed_e,
                                            seed_t
                                            )
        render_ts += new_ts
        render_es += new_es
        for j in range(views_per_candidate):
            r_name = BaseProtocol.get_r_name(q_name, j + 1, beam_i)            
            r_names.append(r_name)
        print(render_ts, render_es)
        return r_names, render_ts, render_es

    
class Protocol2(BaseProtocol):
    """  
    This protocol keeps the first M predictions, perturbing them N // M times
    """
    def __init__(self, conf, scaler, sampler, protocol_name):
        super().__init__(conf, scaler, sampler, protocol_name)
        self.M = 1
        
    # override
    def resample(self, q_name, pred_t, pred_e, seed_e, seed_t, beam_i=0):
        render_es = []
        render_ts = []
        r_names = []
        
        #### keep previous first M estimates #####
        for i in range(self.M):
            old_t = pred_t[i]  # take first prediction
            old_e = pred_e[i]  # take first prediction

            #### keep previous estimate #####
            render_es.append(old_e)
            render_ts.append(old_t)
            r_names.append(BaseProtocol.get_r_name(q_name, 0, beam_i))
            r_names.append(BaseProtocol.get_r_name(q_name, i, beam_i))
            

        views_per_candidate = self.n_views // self.M - 1
        for i in range(self.M):
            t = pred_t[i]  
            e = pred_e[i]  
            new_ts, new_es = self.sampler.sample_batch(views_per_candidate, self.center_std, self.max_angle, 
                                                                     t, e, seed_e, seed_t)
            render_ts += new_ts
            render_es += new_es
            for j in range(views_per_candidate):
                r_name = BaseProtocol.get_r_name(q_name, self.M+i*views_per_candidate+j, beam_i)
                r_names.append(r_name)
        return r_names, render_ts, render_es
