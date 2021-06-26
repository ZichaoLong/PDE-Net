%%
% digits()
syms u u_x u_y u_xx u_xy u_yy v v_x v_y v_xx v_xy v_yy
channels = [u u_x u_y u_xx u_xy u_yy v v_x v_y v_xx v_xy v_yy];
channels = [sym(1,'d') channels].';
for k=1:2
    weight = sym(randn(2,12+k),'d');
    bias = sym(randn(2,1),'d');
    o = weight*channels+bias;
    channels = [channels;o(1)*o(2)];
end
o = expand(sum(channels));
[c,t] = coeffs(o);
% c = double(c);
% t = string(t);
size(c)
