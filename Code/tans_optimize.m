function tans_optimize(TargetNetwork,AvoidanceRegion,PercentileThresholds,SearchGrid,SkinFile,VertexSurfaceArea,Uncertainty,OutDir,normE_Dir)

rng(44); % for reproducibility;
warning ('off','all'); % turn off annoying warnings; users could comment this line out if they want. 
SmoothingFactor = 0.85; % might need to increase if the search grid is very sparse.

% make the optimize dir.
mkdir([OutDir '/Optimize']);

% SkinFile == path to skin surface file;
% Skin == loaded version
Skin = gifti(SkinFile); 

% evaluate E-fields associated with all the coil placements;

% load the search 
% grid metric file;
G = gifti(SearchGrid); 
SearchGridVertices = find(G.cdata~=0); % 

% read the first file & count how many orientations were attempted
files = dir(fullfile(normE_Dir, 'CoilCenter_*', '*_normE.dtseries.nii'));
ids = zeros(numel(files),1);
for i = 1:numel(files)
    tokens = regexp(files(i).name, 'CoilCenter_(\d+)', 'tokens');
    ids(i) = str2double(tokens{1}{1});
end

[~, order] = sort(ids);   % ids = [1 2 3 4 ...]
files = files(order);

MagnE = ft_read_cifti_mod(fullfile(files(1).folder, files(1).name));
nCols = size(MagnE.data,2);

% preallocate;
OnTarget = zeros(length(SearchGridVertices),nCols,length(PercentileThresholds)); % "OnTaget" variable (% of E-field hotspot that contains target network vertices);
Penalty = zeros(length(SearchGridVertices),nCols,length(PercentileThresholds)); % "Penalty" variable (% of E-field hotspot that contains avoidance region / network vertices);

% if no avoidance 
% region is specified; 
if isempty(AvoidanceRegion)
AvoidanceRegion = TargetNetwork; % preallocate
AvoidanceRegion.data = zeros(size(TargetNetwork.data,1), 1);
end

% sweep the search space;
for i = 1:length(files)
   
    % read in the CIFTI file;
    MagnE = ft_read_cifti_mod(fullfile(files(i).folder, files(i).name));

    % make sure we have the correct point in the space grid;
    tmp = strsplit(files(i).name,{'_','.'}); % note: sometimes a given point in the search grid will fail;
    idx = str2double(tmp{2});
    
    % sweep the coil orientations;
    for ii = 1:size(MagnE.data,2)
        
        % sweep all of the thresholds;
        for iii = 1:length(PercentileThresholds)
            
            HotSpot = MagnE.data(1:59412,ii) > prctile(MagnE.data(1:59412,ii),PercentileThresholds(iii)); % this is the hotspot
            OnTarget(idx,ii,iii) = (sum(VertexSurfaceArea.data(HotSpot&TargetNetwork.data(1:59412,1)==1)) / sum(VertexSurfaceArea.data(HotSpot))); 
            Penalty(idx,ii,iii) = (sum(VertexSurfaceArea.data(HotSpot&AvoidanceRegion.data(1:59412,1)==1)) / sum(VertexSurfaceArea.data(HotSpot)));
            
        end
        
    end

    % clear 
    % variable
    clear MagnE
    
end

% save some variables;
save([OutDir '/Optimize/CoilCenter_OnTarget'],'OnTarget');
save([OutDir '/Optimize/CoilCenter_Penalty'],'Penalty');

% average accross the e-field thresholds;
AvgOnTarget = mean(OnTarget,3); % on-target;
AvgPenalty = mean(Penalty,3); % penalty;
AvgPenalizedOnTarget = mean(OnTarget,3) - mean(Penalty,3); % on-target - penalty; relative on-target value used for optimization

G_OnTarget = G; % preallocate;
G_OnTarget.cdata = zeros(size(G_OnTarget.cdata)); % blank slate;

% write out the on-target metric file;
G_OnTarget.cdata(SearchGridVertices) = max(AvgOnTarget,[],2); % average across orientations, for now.
save(G_OnTarget,[OutDir '/Optimize/CoilCenter_OnTarget.shape.gii']); % write out the on-target metric file;
system(['wb_command -metric-smoothing ' SkinFile ' ' OutDir '/Optimize/CoilCenter_OnTarget.shape.gii ' num2str(SmoothingFactor) ' ' OutDir '/Optimize/CoilCenter_OnTarget_s' num2str(SmoothingFactor) '.shape.gii -fix-zeros']);
G_OnTarget = gifti([OutDir '/Optimize/CoilCenter_OnTarget_s' num2str(SmoothingFactor) '.shape.gii']); % read in the smoothed file;

% write out the penalty metric file;
G_Penalty = G_OnTarget; % preallocate
G_Penalty.cdata = zeros(size(G_OnTarget.cdata)); % blank slate;
G_Penalty.cdata(SearchGridVertices) = max(AvgPenalty,[],2); % average across orientations, for now.
save(G_Penalty,[OutDir '/Optimize/CoilCenter_Penalty.shape.gii']); % write out the on-target metric file;
system(['wb_command -metric-smoothing ' SkinFile ' ' OutDir '/Optimize/CoilCenter_Penalty.shape.gii ' num2str(SmoothingFactor) ' ' OutDir '/Optimize/CoilCenter_Penalty_s' num2str(SmoothingFactor) '.shape.gii -fix-zeros']);

% write out the penalized on-target metric file;
G_PenalizedOnTarget = G_OnTarget; % preallocate
G_PenalizedOnTarget.cdata = zeros(size(G_OnTarget.cdata)); % blank slate;
G_PenalizedOnTarget.cdata(SearchGridVertices) = max(AvgPenalizedOnTarget,[],2); % average across orientations, for now.
save(G_PenalizedOnTarget,[OutDir '/Optimize/CoilCenter_PenalizedOnTarget.shape.gii']); % write out the relative on-target metric file;
system(['wb_command -metric-smoothing ' SkinFile ' ' OutDir '/Optimize/CoilCenter_PenalizedOnTarget.shape.gii ' num2str(SmoothingFactor) ' ' OutDir '/Optimize/CoilCenter_PenalizedOnTarget_s' num2str(SmoothingFactor) '.shape.gii -fix-zeros']);
G_PenalizedOnTarget = gifti([OutDir '/Optimize/CoilCenter_PenalizedOnTarget_s' num2str(SmoothingFactor) '.shape.gii']); % read in the smoothed file;

% preallocate the overall quality (adjusted for some amoutn of error
% anticipated from neuronavigation imprecision).
PenalizedOnTarget_ErrorAdjusted = zeros(length(SearchGridVertices),1);

% sweep the search grid vertices;
for i = 1:length(SearchGridVertices)
    D = pdist2(Skin.vertices,Skin.vertices(SearchGridVertices(i),:));
    PenalizedOnTarget_ErrorAdjusted(i) = mean(G_PenalizedOnTarget.cdata(D <= Uncertainty)); % average value of all vertices within the specified distance of vertex "i" 
end

% define the best coil center placement;
[~, order] = sort(PenalizedOnTarget_ErrorAdjusted, 'descend');
TopK = 5;
Idx = order(1:TopK);

OutputData = zeros(TopK, 5); 

for k = 1:TopK
    CoilCenterVertex = SearchGridVertices(Idx(k));
    CoilCenterCoords = Skin.vertices(CoilCenterVertex,:);

    PenalizedPercent = PenalizedOnTarget_ErrorAdjusted(Idx(k)) * 100;

    OutputData(k,:) = [Idx(k),CoilCenterCoords, PenalizedPercent];
end

T = array2table(OutputData, 'VariableNames', ...
    {'CoilCenter_Idx', 'X', 'Y', 'Z', 'OnTarget(%)'});

csvFile = fullfile(OutDir, 'Optimize', 'CoilCenterCoordinates.csv');
writetable(T, csvFile);

end





