data = readmatrix("logistic.txt");


plot(data(:,1),data(:,2),"r","DisplayName","MPGOS Simulation")
legend;
hold on;

xVals = linspace(0,10,1000);
yVals = 2*xVals ./ (xVals + 1);

plot(xVals,yVals,"b--","DisplayName","Analitic Solution")
