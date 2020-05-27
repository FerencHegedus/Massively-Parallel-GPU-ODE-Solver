function PressureReliefValveTestBifurcationdiagram

Data = dlmread('PressureReliefValve.txt');

f=figure(2); hold on;
    set(gca,'YLim',[0 10],'XGrid','on','YGrid','on','Box','on');
p=plot(Data(:,1),Data(:,2));
    set(p,'LineStyle','none','Marker','.','MarkerSize',0.5);
p=plot(Data(:,1),Data(:,3));
    set(p,'LineStyle','none','Marker','.','MarkerSize',0.5);

% Data(1:10,3)