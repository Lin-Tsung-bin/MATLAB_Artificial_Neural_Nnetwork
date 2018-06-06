clc
clear all

net=newp([-2 2;-2 2],1);
net.trainParam.epochs=10;

p1=[[-1; 1] [-1 ;-1] [0; 0] [1; 0]];
t1=[1 1 0 0];
net=train(net,p1,t1);
w=net.IW{1,1}
b=net.b{1}
a=sim(net,p1)
error=t1-a
plotpv(p1,t1);
plotpc(net.IW{1,1},net.b{1});


figure

 p2=[[-2;0] [1;1] [0;1] [-1;-2] [-1;0] [-1;2]];
 a2=sim(net,p2)
 plotpv(p2,a2);
 plotpc(net.IW{1,1},net.b{1});