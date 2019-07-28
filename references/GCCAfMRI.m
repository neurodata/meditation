function Data=GCCAfMRI(numViews,rankTol)
%function Data=GCCAfMRI(numViews,rankTol)
% Perform SUMCOR variant of generalized canonical correlation analysis for 
% fMRI meditation data from Daniel Margulies,
%  Research Group for Neuroanatomy & Connectivity
% International Max Planck Research School
% Code by 
% John M. Conroy IDA/CCS
% June 2019


% Read in fMRI data from subjects correpsonding ignoring those in the
% remove list. The data are normalized across time and the mean value
% of the voxels is subtracted an saved.  

%Subjects to remove: sub-026 037 038 054 050 060 078 022 026 036 037 090 096 101 105 106 10
remove=[];%[026 037 038 054 050 060 078 022 036 090 096 101 105 106 109];  

dataDir='C:\\Users\\Ronan Perry\\Documents\\JHU\\jovo-lab\\meditation\\data\\raw\\';
%Compute canonical correlates for each state
states={'compassion','restingstate','openmonitoring'};
for s=1:length(states)
    Data.(states{s})=loadfMRI(dataDir,remove,states{s},numViews,rankTol);
    numViews=min(numViews,length(Data.(states{s})));
    %Now perform SUMCOR GCCA
    Data.(states{s})=svdgcca(Data.(states{s}),1:numViews);
    break
end

% The structure array Data will have for the i-th subject:
%Data(i).file the file namme where the data was loaded.
%Data(i).X
%Data(i).mean2 the temporal mean 
%Data(i).

function Data=loadfMRI(dur,remove,label,numFiles,rankTol)
%function Data=loadfMRI(dur,remove,label,numFiles,rankTol)
%Load fMRI data and compute rank within rankTol of each via an SVD.
% Note the data are standarized to be mean 0 and variance 1 across
% time and then the column mean (across voxels) is computed and subtracted.


%Select just the files with the given label, e.g. 'compassion',
%'restingstate', or 'openmonitoring'
files=dir([dur,'*',label,'*']);
files=fullfile(dur,{files.name});
%Remove subjects from the remove list
keep=true(size(files));
for i=1:length(files)
    t=regexp(files{i},'sub-(?<index>[0-9][0-9][0-9])','names');
    subIndex=sscanf(t.index,'%d');
    if ismember(subIndex,remove)
        keep(i)=false;
    end
end
files=files(keep);

Data=struct();
for i=1:min(numFiles,length(files))
    Data(i).file=files{i};
    Data(i).X=load(files{i});
    [Data(i).X,Data(i).mu2,Data(i).sigma2]=zscore(Data(i).X,0,2);
    Data(i).mu=mean(Data(i).X);
    Data(i).X=Data(i).X-repmat(Data(i).mu,size(Data(i).X,1),1);
    [Data(i).U,Data(i).S,Data(i).V]=svd(Data(i).X,'econ');
    Data(i).ncols=size(Data(i).X,2);
    Data(i).rank=sum(diag(Data(i).S)>rankTol);
    Data(i).U=Data(i).U(:,1:Data(i).rank);
end

function Data=svdgcca(Data,views)
% Compute SUMCOR canonical coefficients and canonical correlations for the
% given multiview data set. With Data(i).U,Data(i).S,Data(i).V giving
% the SVD of Data(i).X the standardized and column-centered data for the
% i-th view.
% The projected data is stored in Data(i).ProjX.

% 
%Take an SVD of the matrix Uall=[Data(views).U] to compute
%the generalized CCA projections for the set of views
%as seen in the bases Data(i).U for each view i in set of views.

%Form the collected view
Uall=[Data(views).U];
ranks=[Data(views).rank];
d=min(ranks);

fprintf(1,'Computing truncated (rank %d) SVD of size %d by %d via svds\n',...
    d,size(Uall,1),size(Uall,2));
[~,~,VV]=svds(Uall,d);%'econ');
% Note that VV is the matrix of eigenvectors (also singular vectors) of
% Xall=Uall'*Uall, i.e., VV'*Xall*VB is a diagonal matrix of singular values.
% Furthermore, the leading sing valgccaues of Uall are \sqrt{\sigma_i+1} where
%[~,Dall,VV]=svd(Uall'*Uall);
% \sigma_i is the ith sing value of Xall.
n=size(Data(1).X,1);
%ncols=[Data(views).ncols];

VV=VV(:,1:min(d,size(VV,2)));
je=0;
for i=views
    js=je+1;
    je=js+ranks(i)-1;
    %Data(i).A=zeros(ncols,d);
    VVi = normc(VV(js:je,:));
    %VVi=normalize(VV(js:je,:),'norm');
    % Compute the canonical projections
    Data(i).A=sqrt(n-1)*Data(i).V(:,1:ranks(i))*(Data(i).S(1:ranks(i),1:ranks(i))\VVi);
    Data(i).projX=Data(i).X*Data(i).A;
end
%
% The 1 canonical component has median pairwise corr of 0.79
% The 2 canonical component has median pairwise corr of 0.77
% The 3 canonical component has median pairwise corr of 0.72
% The 4 canonical component has median pairwise corr of 0.68
% The 5 canonical component has median pairwise corr of 0.66
% The 6 canonical component has median pairwise corr of 0.65
% The 7 canonical component has median pairwise corr of 0.62
% The 8 canonical component has median pairwise corr of 0.61
% The 9 canonical component has median pairwise corr of 0.60
% The 10 canonical component has median pairwise corr of 0.60
% The 11 canonical component has median pairwise corr of 0.58
% The 12 canonical component has median pairwise corr of 0.58
% The 13 canonical component has median pairwise corr of 0.56
% The 14 canonical component has median pairwise corr of 0.55
% The 15 canonical component has median pairwise corr of 0.55
% The 16 canonical component has median pairwise corr of 0.54
% The 17 canonical component has median pairwise corr of 0.54
% The 18 canonical component has median pairwise corr of 0.54
% The 19 canonical component has median pairwise corr of 0.53
% The 20 canonical component has median pairwise corr of 0.52
% The 21 canonical component has median pairwise corr of 0.51
% The 22 canonical component has median pairwise corr of 0.51
% The 23 canonical component has median pairwise corr of 0.51
% The 24 canonical component has median pairwise corr of 0.51
% The 25 canonical component has median pairwise corr of 0.50
% The 26 canonical component has median pairwise corr of 0.50