data = readmatrix("epidemic.txt");


plot(data(:,1),data(:,2),"r","DisplayName","MPGOS Simulation")
legend;
xlabel("Protection measures \beta","FontSize",14);
ylabel("Maximal number of infected","FontSize",14);

figure(2)
plot(data(:,1),ceil(data(:,3)),"b","DisplayName","MPGOS Simulation")
legend;
xlabel("Protection measures \beta","FontSize",14);
ylabel("Day when maxima reached","FontSize",14);


dense = readmatrix("DenseOutput_0.txt");
figure(3)
plot(dense(:,1),dense(:,2),"g.","DisplayName","MPGOS Simulation")
legend;
xlabel("Days","FontSize",14);
ylabel("Number of infected","FontSize",14);