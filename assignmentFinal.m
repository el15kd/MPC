clear, clc, close all, addpath('References'); % clean & prep
%% State-Space Sys
M=10; D=0.8; RegFact=0.1; Tt=0.5; Tg=0.2; % Model Parameters [A,B,C,D]=tf2ss(b,a)
% continuous s-s model matrices; Bd should be separate from B
A=[-D/M,1/M,0;...
    0,-1/Tt,1/Tt;...
    -1/(RegFact*Tg),0,-1/Tg]; 
Bu=[0;0;1/Tg]; Bd=[-1/M;0;0]; C=[50,0,0]; D=0; tS=0.1;
A=[A,Bu,Bd;...
    zeros(1,3),ones(1,1),0;...
    zeros(1,3),0,ones(1,1)];
B=[Bu;ones(1,1);0]; C=[C,0,D]; D=D;
% ctrllr DT prediction model w/ 0.1 sampling t & zero-order hold sampling
sysc=ss(A,B,C,D,'inputname',{'u'},'outputname','y'); sysd=c2d(sysc,tS); 
Co=ctrb(sysd.A,sysd.B); Ob=obsv(sysd.A,sysd.C);
%d=0.01;
d=-0.3+(0.3-(-0.3))*rand(1); 
% d = pd large step-change in demand up to 0.3 pu
% nxn (1x1) mtrx of uniformly distributed numbers
% generate N random numbers in the interval (a,b) w/ r = a + (b-a).*rand(N,1).
%% Prediction and Cost matrices
n = size(A,1); m = size(Bu,2); % state, input dimension: =2 for B; =1 for Bu
x0=[0.001;0;0]; N=10; %horizon length
[F,G]=predict_mats(A,B,N); Q=C'*C; R=1; P=Q;
%K=-dlqr(A,B,Q,R); 
%KN=K(1:m,:); Acl=A+B*KN; S=(Q+KN'*R*KN); P=dlyap(Acl',S); 
[H,L,M,Qd,Rd]=cost_mats(F,G,Q,R,P); 
%% Checks
unob=length(A)-rank(Ob); % num of unobservable states; if > 0: Ob not FR
unco=length(A)-rank(Co); % num of uncontrollable states; if > 0: Co not FR
stableStatus=isstable(sysd); eigens=eig(sysd); % Verify stability
%eA=eig(Acl); eQ=eig(Q); eP=eig(P);
%% State Predictions
% optimal policy; control sequence; state predictions
x=x0; xi=0; r=0; d=-0.1;
%% Build Constraints
Px=[C;-C]; qx=[0.5;0.5]; Pxf=Px; qxf=[0;0]; Pu=[1;-1]; qu=[0.5;0.5];
[Pc,qc,Sc]=constraint_mats(F,G,Pu,qu,Px,qx,Pxf,qxf);
%% Add Integral Action
Ki=-0.01; Dd=0; uprev=0; r=0; u=uprev; state=[x;u;d];
for k=0:N-1 %Constrained LQ-MPC (RH-LQR)
% the first control in the optimized sequence's applied to sys &
% the optimization problem is subsequently re-solved
    [deltau,jstar,exitFlag]=quadprog(H,L*state,Pc,qc+Sc*state);
    if exitFlag < 1 % reason why quadprog stopped
        disp(['Optimization infeasible at k=' num2str(k)]);     
        break
    end
    u=uprev+deltau(1:m);
    state=A*state+B*deltau(1:m); y=C*state;
    uprev=u; x=state(1:3);
    us(:,k+1)=u; js(:,k+1)=jstar+x'*M(1:3,1:3)*x; xs(:,k+1)=x; ys(:,k+1)=y(1:m);
end
xs=[x0,xs]; round(us,3); ys=round(ys,3); round(js,3); round(xs,3);
figure(1); stairs([0:1:N-1],us);  maxlim=max(abs([us(1) us(end)]));
ylim(1.1*[-maxlim maxlim]); xticks([0:1:N]); xlabel("Time step k"); ylabel("u(k)"); 
box on; title({['Control Inputs over N=',num2str(N)]});
figure(2); plot([0:1:N-1],ys); maxlim=max(abs([ys(1) ys(end)])); box on;
ylim(1.1*[-maxlim maxlim]); xlabel("Time step k"); ylabel("y(k)"); title({['deltaF=y(1) over N=',num2str(N)]});
figure(3); plot([0:1:N-1],ys); 
maxlim=max(abs([js])); box on; %ylim(0.1*[-maxlim maxlim]); 
xlabel("Time step k"); ylabel("y(k)"); title({['Cost J over N=',num2str(N)]});
%{
% compute deadbeat mode-2 K & force state to origin OR origin neighbourhood
%lambdaI=[0.01,0,0;0,0.01,0;0,0,0.01]; 
lambdaI=[zeros(3,3)]; 
lambdaIminusA=lambdaI-A; % A+BK=0; BK=-A; K=B\A
[KdeadBeat, rankKdeadBeat]=linsolve(Bu,lambdaIminusA);
S=(Q+KdeadBeat'*R*KdeadBeat); P=dlyap(Acl',S); % re-compute P for a deadbeat K
[F,G]=predict_mats(A,Bu,N); [H,L,M,Qd,Rd]=cost_mats(F,G,Q,R,P); vect=[];
% Mode-2 constr
for j=1:n
    var=(A+Bu*KdeadBeat); vect=[vect;var]; % 6x3, but want only x(1)->(3x1)
    %var2=KdeadBeat; vect2=[vect2;var2];
end
PxfMode2=(kron(ones(n,n),[Px;Pu*KdeadBeat]))*vect;
qMode2=kron(ones(n,1),[qx;qu]);

[Pc,qc,Sc,~,~,~,~]=constraint_mats(F,G,Pu,qu,Px,qx,PxfMode2,qMode2);
xN=x; % here x=xn
% Extend mode-1 constr to mode-2 & modify Terminal mode-1 constr for mode-2
for j=1:n
    AclPower=(A*x+Bu*KdeadBeat)^(j-1);
    x=(AclPower^(j-1))*xN;
    yPred=C*x; y=yPred+Dd*d; 
    [ustar,jstar,exitFlag]=quadprog(H,L*x,Pc,qc+Sc*x);
        if exitFlag < 1 % reason why quadprog stopped
        disp(['Optimization infeasible at k=' num2str(j)]);     
        break
        end
    us=[us,ustar(1:m)]; 
    js=[js,jstar+x'*M*x];
    ys=[ys,y];
end
%}