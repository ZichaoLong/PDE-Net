import sys
import time
import numpy as np
import torch
import conf,setenv
terms0 = []
terms1 = []
coeffs0 = np.zeros((40))
coeffs1 = np.zeros((40))
options = {}
options['--name'] = sys.argv[1]
configfile = 'checkpoint/'+options['--name']+'/options.yaml'
options = conf.setoptions(argv=None,kw=None,configfile=configfile,isload=True)
options['--device'] = 'cpu'
block = sys.argv[2]

globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

callback.load(int(block))

tsym,csym = model.poly0.coeffs(calprec=8)
terms0 = tsym
coeffs0[:min(len(tsym),40)] = csym[:min(len(tsym),40)]
tsym,csym = model.poly1.coeffs(calprec=8)
terms1 = tsym
coeffs1[:min(len(tsym),40)] = csym[:min(len(tsym),40)]
torch.save(dict(terms0=terms0,terms1=terms1,coeffs0=coeffs0,coeffs1=coeffs1),'coeffs/'+options['--name']+'-block'+str(block))


