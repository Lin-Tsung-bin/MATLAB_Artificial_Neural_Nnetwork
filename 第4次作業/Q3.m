clc 
clear all

p = [-2:.01:2];
t = (1+sin((pi/8)*p))+ 0.1*randn(size(p));
val.P = [-1.975:.01:1.975];
val.T = (1+sin((pi/8)*val.P))+0.1*randn(size(val.P));
net=newff([-1 1],[10,1],{'tansig','purelin'},'trainlm');
net.trainParam.show = 10;
net.trainParam.epochs = 50;
net.trainParam.mu = 1;
net.trainParam.mu_dec = 0.8;
net.trainParam.mu_inc = 1.5;
[net,tr]=train(net,p,t,[],[],val);
a=sim(net,p);

figure(1)
[m,b,r] = postreg(a,t)
R2=rs_new(1+sin(p*(pi/8)),a)
figure(2)
plot(p,t,'+',p,a,'-',p,1+sin(p*(pi/8)),':')
xlabel('Input')
ylabel('Output')
title('Function Approximation')
legend('Noisy sine function','Network response with early stopping',...
       'sine function') 
