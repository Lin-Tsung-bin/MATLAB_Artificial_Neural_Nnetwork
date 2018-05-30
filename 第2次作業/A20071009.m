clear all
clc
net=newlin([-1 1],1,[0 1],0.01);
net.IW{1,1}=[0 0];
net.b{1}=0;
net.trainParam.epochs=500;
Pi={1};
P={-5 -4 -3 -2 -1 0 1 2 3 4 5};
T={-1 -1 -1 -1 -1 -1 -1 1 1 1 1};
net=train(net,P,T,Pi);
a=net.IW{1,1}
e=net.b{1}

p1={-20,-18,-1,2,3,5,30,17,6,-20};
A=sim(net,p1)

for i=1:10
    if A{i}<0
        Y(i)=-1;
    else
        Y(i)=1;
    end
end
Y



%-------------------------------------------------
net=newlin([-1 1],1,[0 1],0.1);
net.IW{1,1}=[0 0];
net.b{1}=0;
% net.inputWeights{1,1}.learnParam.lr=0.1;
% net.biases{1,1}.learnParam.lr=0.1;
% net.trainParam.epochs=100;
Pi={1};
P={-5 -4 -3 -2 -1  0  1 2 3 4 5};
T={-1 -1 -1 -1 -1 -1 -1 1 1 1 1};
net.inputWeights{1,1}.learnParam.lr=0.1;
net.biases{1,1}.learnParam.lr=0.1;
net.trainParam.epochs=1000;
[net,a,e,pf]=adapt(net,P,T,Pi);
a2=net.IW{1,1}
e2=net.b{1}

B=sim(net,p1)

for i=1:10
    if B{i}<0
        Y2(i)=-1;
    else
        Y2(i)=1;
    end
end
Y2