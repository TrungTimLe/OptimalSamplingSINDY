function dx = coupled_vdp(t,x,par)
mu1 = par(1); mu2 = par(2); tau1 = par(3); tau2 = par(4);
c1 = par(5); c2 = par(6);
dx = [...
(x(2) + c1*x(3))/tau1... 
(mu1*(1-x(1)^2)*x(2) - x(1))/tau1...
(x(4) + c2*x(1))/tau2...
(mu2*(1-x(3)^2)*x(4) - x(3))/tau2]';