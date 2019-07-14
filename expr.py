import numpy as np
import torch
import sympy
ISINSTALLMATLAB = True
try:
    import matlab
except ModuleNotFoundError:
    ISINSTALLMATLAB = False
    matlab = None

__all__ = ['poly',]

class poly(torch.nn.Module):
    def __init__(self, hidden_layers, channel_num, channel_names=None, normalization_weight=None):
        super(poly, self).__init__()
        self.hidden_layers = hidden_layers
        self.channel_num = channel_num
        if channel_names is None:
            channel_names = list('u'+str(i) for i in range(self.channel_num))
        self.channel_names = channel_names
        layer = []
        for k in range(hidden_layers):
            module = torch.nn.Linear(channel_num+k,2).to(dtype=torch.float64)
            module.weight.data.fill_(0)
            module.bias.data.fill_(0)
            self.add_module('layer'+str(k), module)
            layer.append(self.__getattr__('layer'+str(k)))
        module = torch.nn.Linear(channel_num+hidden_layers, 1).to(dtype=torch.float64)
        module.weight.data.fill_(0)
        module.bias.data.fill_(0)
        self.add_module('layer_final', module)
        layer.append(self.__getattr__('layer_final'))
        self.layer = tuple(layer)
        nw = torch.ones(channel_num).to(dtype=torch.float64)
        if (not isinstance(normalization_weight, torch.Tensor)) and (not normalization_weight is None):
            normalization_weight = np.array(normalization_weight)
            normalization_weight = torch.from_numpy(normalization_weight).to(dtype=torch.float64)
            normalization_weight = normalization_weight.view(channel_num)
            nw = normalization_weight
        self.register_buffer('_nw', nw)
    @property
    def channels(self):
        channels = sympy.symbols(self.channel_names)
        return channels
    def renormalize(self, nw):
        if (not isinstance(nw, torch.Tensor)) and (not nw is None):
            nw = np.array(nw)
            nw = torch.from_numpy(nw)
        nw1 = nw.view(self.channel_num)
        nw1 = nw1.to(self._nw)
        nw0 = self._nw
        scale = nw0/nw1
        self._nw.data = nw1
        for L in self.layer:
            L.weight.data[:,:self.channel_num] *= scale
        return None
    def _cast2numpy(self, layer):
        weight,bias = layer.weight.data.cpu().numpy(), \
                layer.bias.data.cpu().numpy()
        return weight,bias
    def _cast2matsym(self, layer, eng):
        weight,bias = self._cast2numpy(layer)
        weight,bias = weight.tolist(),bias.tolist()
        weight,bias = matlab.double(weight),matlab.double(bias)
        eng.workspace['weight'],eng.workspace['bias'] = weight,bias
        eng.workspace['weight'] = eng.eval("sym(weight,'d')")
        eng.workspace['bias'] = eng.eval("sym(bias,'d')")
        return None
    def _cast2symbol(self, layer):
        weight,bias = self._cast2numpy(layer)
        weight,bias = sympy.Matrix(weight),sympy.Matrix(bias)
        return weight,bias
    def _sympychop(self, o, calprec):
        cdict = o.expand().as_coefficients_dict()
        o = 0
        for k,v in cdict.items():
            if abs(v)>0.1**calprec:
                o = o+k*v
        return o
    def _matsymchop(self, o, calprec, eng):
        eng.eval('[c,t] = coeffs('+o+');', nargout=0)
        eng.eval('c = double(c);', nargout=0)
        eng.eval('c(abs(c)<1e-'+calprec+') = 0;', nargout=0)
        eng.eval(o+" = sum(sym(c, 'd').*t);", nargout=0)
        return None
    def expression(self, calprec=6, eng=None, isexpand=True):
        if eng is None:
            channels = sympy.symbols(self.channel_names)
            for i in range(self.channel_num):
                channels[i] = self._nw[i].item()*channels[i]
            channels = sympy.Matrix([channels,])
            for k in range(self.hidden_layers):
                weight,bias = self._cast2symbol(self.layer[k])
                o = weight*channels.transpose()+bias
                if isexpand:
                    o[0] = self._sympychop(o[0], calprec)
                    o[1] = self._sympychop(o[1], calprec)
                channels = list(channels)+[o[0]*o[1],]
                channels = sympy.Matrix([channels,])
            weight,bias = self._cast2symbol(self.layer[-1])
            o = (weight*channels.transpose()+bias)[0]
            if isexpand:
                o = o.expand()
                o = self._sympychop(o, calprec)
            return o
        else:
            calprec = str(calprec)
            eng.clear(nargout=0)
            eng.syms(self.channel_names, nargout=0)
            channels = ""
            for c in self.channel_names:
                channels = channels+" "+c
            eng.eval('syms'+channels,nargout=0)
            channels = "["+channels+"].'"
            eng.workspace['channels'] = eng.eval(channels)
            eng.workspace['nw'] = matlab.double(self._nw.data.cpu().numpy().tolist())
            eng.eval("channels = channels.*nw.';", nargout=0)
            for k in range(self.hidden_layers):
                self._cast2matsym(self.layer[k], eng)
                eng.eval("o = weight*channels+bias';", nargout=0)
                eng.eval('o = o(1)*o(2);', nargout=0)
                if isexpand:
                    eng.eval('o = expand(o);', nargout=0)
                    self._matsymchop('o', calprec, eng)
                eng.eval('channels = [channels;o];', nargout=0)
            self._cast2matsym(self.layer[-1],eng)
            eng.eval("o = weight*channels+bias';", nargout=0)
            if isexpand:
                eng.eval("o = expand(o);", nargout=0)
                self._matsymchop('o', calprec, eng)
            return eng.workspace['o']
    def coeffs(self, calprec=6, eng=None, o=None, iprint=0):
        if eng is None:
            if o is None:
                o = self.expression(calprec, eng=None, isexpand=True)
            cdict = o.as_coefficients_dict()
            t = np.array(list(cdict.keys()))
            c = np.array(list(cdict.values()), dtype=np.float64)
            I = np.abs(c).argsort()[::-1]
            t = list(t[I])
            c = c[I]
            if iprint > 0:
                print(o)
        else:
            if o is None:
                self.expression(calprec, eng=eng, isexpand=True)
            else:
                eng.workspace['o'] = eng.expand(o)
            eng.eval('[c,t] = coeffs(o);', nargout=0)
            eng.eval('c = double(c);', nargout=0)
            eng.eval("[~,I] = sort(abs(c), 'descend'); c = c(I); t = t(I);", nargout=0)
            eng.eval('m = cell(numel(t),1);', nargout=0)
            eng.eval('for i=1:numel(t) m(i) = {char(t(i))}; end', nargout=0)
            if iprint > 0:
                eng.eval('disp(o)', nargout=0)
            t = list(eng.workspace['m'])
            c = np.array(eng.workspace['c']).flatten()
        return t,c
    def symboleval(self,inputs,eng=None,o=None):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.data.cpu().numpy()
        if isinstance(inputs, np.ndarray):
            inputs = list(inputs)
        assert len(inputs) == len(self.channel_names)
        if eng is None:
            if o is None:
                o = self.expression()
            return o.subs(dict(zip(self.channels,inputs)))
        else:
            if o is None:
                o = self.expression(eng=eng)
            channels = "["
            for c in self.channel_names:
                channels = channels+" "+c
            channels = channels+"].'"
            eng.workspace['channels'] = eng.eval(channels)
            eng.workspace['tmp'] = o
            eng.workspace['tmpv'] = matlab.double(inputs)
            eng.eval("tmpresults = double(subs(tmp,channels.',tmpv));",nargout=0)
            return np.array(eng.workspace['tmpresults'])
    def forward(self, inputs):
        outputs = inputs*self._nw
        for k in range(self.hidden_layers):
            o = self.layer[k](outputs)
            outputs = torch.cat([outputs,o[...,:1]*o[...,1:]], dim=-1)
        outputs = self.layer[-1](outputs)
        return outputs[...,0]
