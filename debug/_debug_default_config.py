import numpy as np
def default_options():
    options = {
            # computing region
            '--eps':2*np.pi,
            '--cell_num':1,
            # super parameters of network
            '--kernel_size':7,
            '--max_order':2,
            '--dx':2*np.pi/32,
            # data generator
            '--viscosity':0.05,
            '--freq':4,
            # data transform
            '--start_noise':0.0,
            '--end_noise':0.0,
            }
    return options
