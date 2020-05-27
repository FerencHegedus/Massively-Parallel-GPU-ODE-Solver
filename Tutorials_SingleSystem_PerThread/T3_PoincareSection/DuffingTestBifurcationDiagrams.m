function DuffingTestBifurcationDiagrams

FigureNumber = 1;
Data = dlmread('DuffingPoincare.txt');

% First component of the Poincaré section
figure(FigureNumber+1);
p1=plot(Data(:,1),Data(:,3));
set(p1,'LineStyle','none','Marker','.','MarkerSize',0.5);

% Second component of the Poincaré section
figure(FigureNumber+2);
p1=plot(Data(:,1),Data(:,4));
set(p1,'LineStyle','none','Marker','.','MarkerSize',0.5);