f_r = zeros(1,11);
r = 0: 1 : 10;
N_max = 10;
P_max = 10;
f_r = 1.9.^r*exp(-1.9)./factorial(r)
plot(r, f_r)