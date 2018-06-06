clc
clear
%n=-5:.1:5
%plot(n,hardlims(n-2))
%a
net=newlin([-1 1],1,[0 1],0.01);
net.iw{1,1}=[0 0];
net.b{1}=0;
net.trainparam.epochs=1000;
pi={1};
p={-5 -4 -3 -2 -1 0 1 2 3 4 5};
t={-1 -1 -1 -1 -1 -1 -1 1 1 1 1};
net=train(net,p,t,pi);
IW = net.iw{1,1}
b = net.b{1}
ps={-20,-18,-1,2,3,5,30,17,6,-20};
A=sim(net,ps);

for i=1:10
    if A{i}<0
        Y(i)=-1;
    else
        Y(i)=1;
    end
end
Y

%repeats = true;
%quitnow = 'x';
%while repeats
%    s=input('Enter text number or x to quit:','s');
%    if s == quitnow
%        break
%    else
%        A=sim(net, str2num(s));
%        if A<0
%            Y=-1
%        else
%            Y=1
%        end
%    end
%end


%b
net=newlin([-1 1],1,[0 1],0.01);
net.iw{1,1}=[0 0];
net.b{1}=0;
net.inputWeights{1,1}.learnParam.lr=0.01;
net.biases{1,1}.learnParam.lr=0.01;
p={[-5] [-4] [-3] [-2] [-1] [0] [1] [2] [3] [4] [5]};
t={-1 -1 -1 -1 -1 -1 -1 1 1 1 1};
for i=1:5
    [net,a,e,pf]=adapt(net,p,t);
end
IW = net.iw{1,1}
b = net.b{1}
B=sim(net,ps);

for i=1:10
    if B{i}<0
        Y(i)=-1;
    else
        Y(i)=1;
    end
end
Y
