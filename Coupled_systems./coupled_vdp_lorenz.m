function dx = coupled_vdp_lorenz(t,x,par_vec)
mu = par_vec(1); sigma = par_vec(2); rho = par_vec(3);
beta = par_vec(4); tau1 = par_vec(5); tau2 = par_vec(6);
c1 = par_vec(7); c2 = par_vec(8);
dx = [...
(x(2) + c1*x(3))/tau1...
(mu*(1-x(1)^2)*x(2) - x(1))/tau1...
(sigma*(x(4) - x(3)) + c2*x(1))/tau2...
(x(3)*(rho - x(5)) - x(4))/tau2...
(x(3)*x(4) - beta*x(5))/tau2]';