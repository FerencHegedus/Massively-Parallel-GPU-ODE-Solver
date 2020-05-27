function KellerMiksis_1f_Plot

Data = dlmread('KellerMiksis.txt');

f=figure(10);
p=plot(Data(:,2),Data(:,5));
    set(p,'LineStyle','none','Marker','.','MarkerSize',6,'Color',[0 0 0]);
    set(gca,'XLim',[20 1000],'XScale','log','XGrid','on','YGrid','on','Box','on','FontSize',14,'FontWeight','bold','FontName','Times');
    
    xlabel('f (kHz)','FontSize',20,'FontWeight','bold','FontName','Times');
    ylabel('y_1^{max}','FontSize',20,'FontWeight','bold','FontName','Times');