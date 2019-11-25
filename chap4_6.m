%CMAC and PID Concurrent Control
clear all;
close all;

ts=0.001;                      %����ʱ����ts=0.001
sys=tf(1770,[1,60,1770]);      %ϵͳ���ݺ���
dsys=c2d(sys,ts,'z');          %������ʱ��ģ��ת������ɢ��ʱ��ģ��
[num,den]=tfdata(dsys,'v');    % numΪ����ϵ����denΪ��ĸϵ�������Z���ݺ����ķ��Ӻͷ�ĸ

alfa=0.04;                     %����ϵ��

N=500;C=5;                     %NΪ����ϵ��  CΪ��������
w=zeros(N+C,1);
w_1=w;w_2=w;d_w=w;

y_1=0;y_2=0;y_3=0;
u_1=0.0;u_2=0.0;u_3=0.0;

x=[0,0,0]';
error_1=0;

s=3; %  Selecting Signal
if s==1  %Sine Signal
    A=0.50;
    Smin=-A;
    Smax=A;
    xite =0.10;        %ѧϰ��
    kp=25;
    ki=0.0;
    kd=0.28;
 
elseif s==2  % Square Wave Signal
    A=0.50;
    Smin=-A;
    Smax=A;
    xite =0.10;
    kp=25;
    ki=0.0;
    kd=0.28;
elseif s==3  %Step Signal
    Smax=1.0;
    Smin=0;
    xite =0.10;
    kp=25;
    ki=0.0;
    kd=0.28;  
elseif s==4  %Random Signal
    A=1.80;
    Smin=-A;
    Smax=A;
    xite =0.10;
    kp=20;
    ki=0.0;
    kd=0.28; 
end

%Coding Input Value  ��������
dvi=(Smax-Smin)/(N-1);            % N-1 ��ֹ����Ϊ0

for i=1:1:C     %C size
    v(i)=Smin;
end

for i=C+1:1:C+N  % N size
    v(i)=v(i-1)+dvi;
end
for i=N+C+1:N+2*C  %C size
    v(i)=Smax;
end
 
for k=1:1:200
    time(k)=k*ts;   
    
    if s==1
        rin(k)=A*sin(4*2*pi*k*ts);         % Sin Signal
    elseif s==2
        rin(k)=A*sign(sin(2*2*pi*k*ts));   % Square  Signal  �����ź�
    elseif s==3
        rin(k)=1;                          %Step Signal
    elseif s==4
        rin(k)=1.0*sin(2*2*pi*k*ts)+0.5*sin(2*pi*5*k*ts)+0.3*sin(2*pi*7*k*ts);
    end
    
    for i=1:1:N+C
        if rin(k)>=v(i)&&rin(k)<=v(i+C)
            a(i)=1;
        else
            a(i)=0;
        end
    end
    
    yout(k)=-den(2)*y_1-den(3)*y_2+num(2)*u_1+num(3)*u_2;        %������
    error(k)=rin(k)-yout(k);
    
    %CMAC Neural Network Controller 
    un(k)=a*w;
    
    %PID controller
    up(k)=kp*x(1)+kd*x(2)+ki*x(3);
    
    MM= 2;         %ģʽѡ��
    if MM==1      %Only using PID Control
        u(k)= up(k);
    elseif MM==2   %Total control output
        u(k)=up(k)+un(k);
    end
    if k==150  %Disturbance  ��Ϊ����
        u(k)=u(k)+5.0;
    end
    %����޷�
    if u(k)>=10  
        u(k)=10;
    end
    if u(k)<=-10
        u(k)=-10;
    end
    % Update NN Weight  ��������Ȩֵ
    
    if s==1|s==4
        d_w=xite*(u(k)-un(k))/C;     %Sin Signal
    elseif s==2|s==3
        d_w=a'*xite*(u(k)-un(k))/C;  %Step Signal and Square Signal
    end
    
    w=w_1+d_w+alfa*(w_1-w_2);
    
    %Parameters  Update ��������
    w_3=w_2;w_2=w_1;w_1=w;
    u_2=u_1;u_1=u(k);
    y_2=y_1;y_1=yout(k);
    
    x(1)=error(k);               % Calculating P  ���� P
    x(2)=(error(k)-error_1)/ts;  % Calculating D  ���� D
    x(3)=x(3)+error(k)*ts;       % Calculating I  ���� I
    
    error_2=error_1;error_1=error(k);
end
 
figure(1);
plot(time,rin,'b',time,yout,'r');
xlabel('time(s)');ylabel('(rin and y)');
figure(2);
subplot(311);
plot(time,un);
xlabel('time(s)');ylabel('un');
subplot(312);
plot(time,up);
xlabel('time(s)');ylabel('up');
subplot(313);
plot(time,u);
xlabel('time(s)');ylabel('u');
figure(3);
plot(time,error,'r');
xlabel('time(s)');ylabel('error');
   
        
            
        
        