function DuffingTestDenseOutput
global k B

ThreadID = 0;

k  = dlmread( strcat('DenseOutput_',num2str(ThreadID),'.txt'), ',', [1 0 1 0]);
B  = dlmread( strcat('DenseOutput_',num2str(ThreadID),'.txt'), ',', [3 0 3 0]);

Data  = dlmread( strcat('DenseOutput_',num2str(ThreadID),'.txt'), ',', 12, 0);

TimeDomain=[0 2*pi];
InitialCondition=[Data(1,2) Data(1,4)];
options = odeset('RelTol',1e-9,'AbsTol',1e-9,'InitialStep',1e-2,'Stats','on');
    [T,Y] = ode45(@OdeFunction,TimeDomain,InitialCondition,options);

disp(Y(end,1))

figure(1); hold on;
plot(T,Y(:,1));
plot(Data(:,1),Data(:,2));

function dy = OdeFunction(t,y)
global k B

dy=zeros(2,1);

dy(1) = y(2);
dy(2) = y(1) - y(1).^3 - k*y(2) + B*cos(t);