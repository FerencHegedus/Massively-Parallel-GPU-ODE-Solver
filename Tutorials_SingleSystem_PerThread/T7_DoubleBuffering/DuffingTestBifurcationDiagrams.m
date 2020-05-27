function DuffingTestBifurcationDiagrams

FigureNumber = 1;
Data = dlmread('Duffing.txt');

% First component of the Poincaré section
figure(FigureNumber+1);
p1=plot(Data(:,1),Data(:,3));
set(p1,'LineStyle','none','Marker','.','MarkerSize',0.5);

% Second component of the Poincaré section
figure(FigureNumber+2);
p1=plot(Data(:,1),Data(:,4));
set(p1,'LineStyle','none','Marker','.','MarkerSize',0.5);

% Local maxima determined by event handling
figure(FigureNumber+3);
p2=plot(Data(:,1),Data(:,5));
set(p2,'LineStyle','none','Marker','.','MarkerSize',0.5);

% Global maxima determined by accessories
figure(FigureNumber+4);
p2=plot(Data(:,1),Data(:,7));
set(p2,'LineStyle','none','Marker','.','MarkerSize',0.5);

% Second component of the solution at the secondly detected second event function
figure(FigureNumber+5);
p2=plot(Data(:,1),Data(:,6));
set(p2,'LineStyle','none','Marker','.','MarkerSize',0.5);