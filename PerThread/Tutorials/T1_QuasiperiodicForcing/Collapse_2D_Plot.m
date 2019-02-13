function Collapse_2D_BifurcationPlot_v8p0

Parameters.PressureAmplitude1 = 1.1;
Parameters.PressureAmplitude2 = 1.2;
Parameters.EquilibriumRadius  = 10;

% Only make bi-parametric plot in the dual-frequency parameter plane

% DIAGRAM OPTIONS
% Relative bubble radius expansion : 'rx1'  [-]
% Compression ratio                : 'crx1' [-]
% Number of Relative bubble radius expansion : 'Nrx1'  [-]
% Number of Compression ratio                : 'Ncrx1' [-]

% Axes directions       : normal,  reverse
% Axes scale            : linear,  log

OperationParameters.Axes           = {'RelativeFrequency1','RelativeFrequency2','rx1'};
OperationParameters.AxesDirections = {'normal','normal','normal'};
OperationParameters.AxesScale      = {'log','log','linear'};

OperationParameters.XRange = 'auto'; % [x y] or 'auto'
OperationParameters.YRange = 'auto'; % [x y] or 'auto'
OperationParameters.ZRange = [0 5]; % [x y] or 'auto'

OperationParameters.Threshold   = 5;
OperationParameters.SaveFigures = 0;


FullFileName = GenerateFilenameWithLocation(Parameters);

GenerateBifurcationDiagram(Parameters,OperationParameters,FullFileName);

%--------------------------------------------------------------------------

function FullFileName = GenerateFilenameWithLocation(Parameters)

PressureAmplitudesString = strcat('PA1_',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_',...
                                  'PA2_',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.txt');
FullFileName = strcat('DataRepository\KellerMiksis_Collapse_',PressureAmplitudesString);


function GenerateBifurcationDiagram(Parameters,OperationParameters,FullFileName)

RawData = dlmread(FullFileName);


BifurcationData.x          = RawData(:,2);
BifurcationData.xAxisLabel = strcat( '\omega_{R1} @P_{A1}=', num2str(Parameters.PressureAmplitude1) );
BifurcationData.y          = RawData(:,4);
BifurcationData.yAxisLabel = strcat( '\omega_{R2} @P_{A2}=', num2str(Parameters.PressureAmplitude2) );

switch OperationParameters.Axes{3}
    case 'rx1'
        BifurcationData.z          = max( RawData(:,9:72), [], 2 ) - 1;
        BifurcationData.zAxisLabel = '(R^{max}-R_E)/R_E';
    case 'Nrx1'
        x1Data = RawData(:,9:72);
        idxThr = x1Data > (OperationParameters.Threshold + 1);
        BifurcationData.z          = sum(idxThr,2) ./ RawData(:,8);
        BifurcationData.zAxisLabel = strcat( 'N( (R^{max}-R_E)/R_E )=', num2str(OperationParameters.Threshold + 1), '/ sec' );
    otherwise
        fprintf('Wrong setup for axis 3!\n');
end


Grid.x = GenerateGridPoints(BifurcationData.x);
if strcmp( OperationParameters.XRange, 'auto') ~= 1
    Idx = ( (Grid.x + 1e-6) < OperationParameters.XRange(1) ) | ( (Grid.x - 1e-6) > OperationParameters.XRange(2) ) ;
    Grid.x(Idx)=[];
end
Grid.y = GenerateGridPoints(BifurcationData.y);
if strcmp( OperationParameters.YRange, 'auto') ~= 1
    Idx= ( (Grid.y + 1e-6) < OperationParameters.YRange(1) ) | ( (Grid.y - 1e-6) > OperationParameters.YRange(2) ) ;
    Grid.y(Idx)=[];
end

[Mesh.x, Mesh.y]= meshgrid(Grid.x, Grid.y);


Values = SortData(Grid.x,Grid.y,BifurcationData,OperationParameters);
fprintf('Valuse extends: %6.4f to %6.4f\n', max(max(Values)), min(min(Values)) );


PlotValues(Mesh.x, Mesh.y, Values, Parameters, OperationParameters, ...
           BifurcationData.xAxisLabel, BifurcationData.yAxisLabel, BifurcationData.zAxisLabel);

clear RawData

%--------------------------------------------------------------------------

function Grid=GenerateGridPoints(Data)

Counter=1;
while length(Data)~=0
    Grid(Counter) = Data(Counter);
    Idx = find( abs( Data(:) - Grid(Counter) ) < 1e-6 );
    Data(Idx)=[];
    Counter = Counter + 1;
end
Grid = sort(Grid);

function Values = SortData(GridX, GridY, BifurcationData, OperationParameters)

Nx = length(GridX);
Ny = length(GridY);

Values = zeros(Ny,Nx);
for i = 1 : Nx
    Idx1 = find( abs( BifurcationData.x - GridX(i) ) < 1e-6 );
    ReducedData.x = BifurcationData.x(Idx1);
    ReducedData.y = BifurcationData.y(Idx1);
    ReducedData.z = BifurcationData.z(Idx1);
    for j = 1 : Ny
        if isempty(Idx1)
            Values(j,i)=0;
        else
            Idx2 = find( abs( ReducedData.y - GridY(j) ) < 1e-6 );
            if isempty(Idx2)
                Values(j,i)=0;
            else
                if length(Idx2)>1
                    fprintf('Wrong number of values!\n');
                else
                    Values(j,i) = ReducedData.z(Idx2);
                end
            end
        end
    end
end

%--------------------------------------------------------------------------

function PlotValues(MeshX, MeshY, Values, Parameters, OperationParameters, xAxisLabel, yAxisLabel, zAxisLabel)

FigureName='3D_Values';
    Fig1=figure('Name',FigureName,'NumberTitle','off','Position',[300 300 1000 550]);

Plot1 = surf(MeshX, MeshY, Values);
    ax1 = Fig1.CurrentAxes;
    set(Plot1,'LineStyle','none');

colorbar(); colormap(jet); caxis( OperationParameters.ZRange );
view(30,50);

set(ax1, 'Box','on', 'XScale',OperationParameters.AxesScale{1}, 'YScale',OperationParameters.AxesScale{2}, 'ZScale',OperationParameters.AxesScale{3});
if strcmp( OperationParameters.ZRange, 'auto') ~= 1
    set(ax1, 'Zlim', [OperationParameters.ZRange(1) OperationParameters.ZRange(2)]);
end
    xlabel(ax1,xAxisLabel, 'FontSize',16, 'FontName','Times', 'FontWeight','bold');
    ylabel(ax1,yAxisLabel, 'FontSize',16, 'FontName','Times', 'FontWeight','bold');
    zlabel(ax1,zAxisLabel, 'FontSize',16, 'FontName','Times', 'FontWeight','bold');

if OperationParameters.SaveFigures==1
    Folder = strcat('DataRepository\Figures\',OperationParameters.SaveDirectory,'\',OperationParameters.SaveExtension,'\','3D\');
    if exist(Folder,'dir')==0, mkdir(Folder); end

    FullFileNameFig = strcat(OperationParameters.Axes{end},'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.fig');
    FullFileNamePNG = strcat(OperationParameters.Axes{end},'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.png');
    if strcmp( OperationParameters.Axes{3}, 'Nx1') == 1
        FullFileNameFig = strcat(OperationParameters.Axes{end},'_thr',num2str(OperationParameters.Threshold),'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.fig');
        FullFileNamePNG = strcat(OperationParameters.Axes{end},'_thr',num2str(OperationParameters.Threshold),'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.png');
    end

    savefig( strcat(Folder, FullFileNameFig) );
    saveas(gcf, strcat(Folder, FullFileNamePNG), 'png');
end

%close;


FigureName='2D_Plot';
    Fig2=figure('Name',FigureName,'NumberTitle','off');

Plot2 = surf(MeshX, MeshY, Values);
    ax2 = Fig2.CurrentAxes;
    set(Plot2,'LineStyle','none');

colorbar(); colormap(jet); caxis( OperationParameters.ZRange );
view(0,90);

set(ax2,'Box','on','XScale', OperationParameters.AxesScale{1}, 'YScale', OperationParameters.AxesScale{2});
    xlabel(ax2,xAxisLabel, 'FontSize',16, 'FontName','Times', 'FontWeight','bold');
    ylabel(ax2,yAxisLabel, 'FontSize',16, 'FontName','Times', 'FontWeight','bold');

if OperationParameters.SaveFigures==1
    Folder = strcat('DataRepository\Figures\',OperationParameters.SaveDirectory,'\',OperationParameters.SaveExtension,'\','2D\');
    if exist(Folder,'dir')==0, mkdir(Folder); end

    FullFileNameFig = strcat(OperationParameters.Axes{end},'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.fig');
    FullFileNamePNG = strcat(OperationParameters.Axes{end},'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.png');
    if strcmp( OperationParameters.Axes{3}, 'Nx1') == 1
        FullFileNameFig = strcat(OperationParameters.Axes{end},'_thr',num2str(OperationParameters.Threshold),'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.fig');
        FullFileNamePNG = strcat(OperationParameters.Axes{end},'_thr',num2str(OperationParameters.Threshold),'_','PA1-',num2str(Parameters.PressureAmplitude1,'%6.2f'),'_PA2-',num2str(Parameters.PressureAmplitude2,'%6.2f'),'.png');
    end

    savefig( strcat(Folder, FullFileNameFig) );
    saveas(gcf, strcat(Folder, FullFileNamePNG), 'png');
end

%close;