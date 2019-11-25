clear all;
close all;

xite=0.25;  % 学习速率
alfa=0.05;  % 惯性系数

S=1;     %信号类型

IN=4;H=5;Out=3;     %网络结构 
if S==1     %阶跃
wi=[-0.6394 -0.2696 -0.3756 -0.7023;
-0.8603 -0.2013 -0.5024 -0.2596;
-1.0749 0.5543 -1.6820 -0.5437;
-0.3625 -0.0724 -0.6463 -0.2859;
0.1425 0.0279 -0.5406 -0.7660];    %隐层初始权值随机[-0.5,0.5]
    %wi=0.50*rands(H,IN);
wi_1=wi;wi_2=wi;wi_3=wi;
wo=[0.7576 0.2616 0.5820 -0.1416 -0.1325;
-0.1146 0.2949 0.8352 0.2205 0.4508;
0.7201 0.4566 0.7672 0.4962 0.3632]; %输出层初始权值随机[-0.5,0.5]
    %wo=0.50*rands(Out,H);
wo_1=wo;wo_2=wo;wo_3=wo;
end

if S==2     %Sine Signal
wi=[-0.2846 0.2193 -0.5097 -1.0668;
-0.7484 -0.1210 -0.4708 0.0988;
-0.7176 0.8297 -1.6000 0.2049;
-0.0858 0.1925 -0.6346 0.0347;
0.4358 0.2369 -0.4564 -0.1324];  
    %wi=0.50*rands(H,IN);
wi_1=wi;wi_2=wi;wi_3=wi;
wo=[1.0438 0.5478 0.8682 0.1446 0.1537;
0.1716 0.5811 1.1214 0.5067 0.7370;
1.0063 0.7428 1.0534 0.7824 0.6494];
    %wo=0.50*rands(Out,H);
wo_1=wo;wo_2=wo;wo_3=wo;
end
%初始化
x=[0,0,0];
u_1=0;u_2=0;u_3=0;u_4=0;u_5=0;
y_1=0;y_2=0;y_3=0;

Oh=zeros(H,1);   %Output from NN middle layer  隐层输出
I = Oh;    %Input to NN middle layer           隐层输入
error_2 = 0;
error_1 = 0;

ts = 0.001;
for k =1:1:6000
time(k)=k*ts;
% 输入量
if S==1  
rin(k)=1.0;   
elseif S==2
rin(k)=sin(1*2*pi*k*ts);
end

%Unlinear model  非线性模型
a(k)=1.2*(1-0.8*exp(-0.1*k));   %a(k)为慢时变
yout(k)=a(k)*4*y_1/(1+y_1^2)+u_1;

error(k)=rin(k)-yout(k);    %误差

xi=[rin(k),yout(k),error(k),1];

x(1)=error(k)-error_1;
x(2)=error(k);
x(3)=error(k)-2*error_1+error_2;

epid=[x(1);x(2);x(3)];
I=xi*wi';    % 隐层输入
for j=1:1:H
Oh(j)=(exp(I(j))-exp(-I(j)))/(exp(I(j))+exp(-I(j))); %Middle Layer  隐层输出
end
K=wo*Oh; %Output Layer  当前输出层权值
for l=1:1:Out
K(l)=exp(K(l))/(exp(K(l))+exp(-K(l))); %Getting kp,ki,kd  激活输出
end
kp(k)=K(1);ki(k)=K(2);kd(k)=K(3);
Kpid=[kp(k),ki(k),kd(k)];

du(k)=Kpid*epid;  % 控制量变化量
u(k)=u_1+du(k);   % 控制输出量
if u(k) >= 20 % Restricting the output of controller  输出限幅
u(k)=20;
end
if u(k)<=-20
u(k)=-20;
end

dyu(k)=sign((yout(k)-y_1)/(u(k)-u_1+0.0000001)); % y(k)变化量

%Output layer     输出层
for j=1:1:Out
dK(j)=2/(exp(K(j))+exp(-K(j)))^2;
end
for l=1:1:Out
delta3(l)=error(k)*dyu(k)*epid(l)*dK(l);
end

for l=1:1:Out
for i=1:1:H
d_wo=xite*delta3(l)*Oh(i)+alfa*(wo_1-wo_2);
end
end
wo=wo_1+d_wo+alfa*(wo_1-wo_2);    %输出层权值更新
%Hidden layer    隐层
for i=1:1:H
dO(i)=4/(exp(I(i))+exp(-I(i)))^2;
end
segma=delta3*wo;
for i=1:1:H
delta2(i)=dO(i)*segma(i);
end

d_wi=xite*delta2'*xi;
wi=wi_1+d_wi+alfa*(wi_1-wi_2);  % 隐层权值更新

    %Parameters Update  参数更新 并 存储历史值
u_5=u_4;u_4=u_3;u_3=u_2;u_2=u_1;u_1=u(k);
y_2=y_1;y_1=yout(k);

wo_3=wo_2;
wo_2=wo_1;
wo_1=wo;

wi_3=wi_2;
wi_2=wi_1;
wi_1=wi;

error_2=error_1;
error_1=error(k);
end
figure(1);
subplot(3, 1,1);
plot(time,rin,'r',time,yout,'b');
xlabel('time(s)');ylabel('rin,yout');
subplot(3, 1,2);
plot(time,error,'r');
xlabel('time(s)');ylabel('error');
subplot(3, 1,3);
plot(time,u,'r');
xlabel('time(s)');ylabel('u');
figure(2);
subplot(311);
plot(time,kp,'r');
xlabel('time(s)');ylabel('kp');
subplot(312);
plot(time,ki,'g');
xlabel('time(s)');ylabel('ki');
subplot(313);
plot(time,kd,'b');
xlabel('time(s)');ylabel('kd');