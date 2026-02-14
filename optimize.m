DataDir = 'Data';
Subject = 'ME01';

% define inputs
VertexSurfaceArea = ft_read_cifti_mod([DataDir '/' Subject '/anat/T1w/fsaverage_LR32k/' Subject '.midthickness_va.32k_fs_LR.dscalar.nii']);
SearchGrid = [DataDir '/' Subject '/tans/Network_Frontoparietal/SearchGrid/SubSampledSearchGrid.shape.gii'];
SkinSurf = [DataDir '/' Subject '/tans/HeadModel/m2m_' Subject '/Skin.surf.gii'];
OutDir = [DataDir '/' Subject '/TMSNet/'];
PercentileThresholds = linspace(99.9,99,10);
Uncertainty = 5;
normE_Dir = [DataDir '/' Subject '/TMSNet/normE'];

FunctionalNetworks = ft_read_cifti_mod([DataDir '/' Subject '/pfm/'...
    Subject '_FunctionalNetworks.dtseries.nii']);
% isolate the target network again
TargetNetwork = FunctionalNetworks;
TargetNetwork.data(TargetNetwork.data~=9) = 0; % note: 9 == frontoparietal network map.
TargetNetwork.data(TargetNetwork.data~=0) = 1; % binarize.

% run the "tans_optimize.m" module;
addpath('Code')
tans_optimize(TargetNetwork,[],PercentileThresholds,SearchGrid,SkinSurf,VertexSurfaceArea,Uncertainty,OutDir,normE_Dir);