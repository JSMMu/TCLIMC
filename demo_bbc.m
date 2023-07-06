close all;
clear;
clc;

addpath('utility');
addpath('data');
addpath('./twist');
datadir = 'data/';
dataname = {'BBC4view_685','20newsgroups'};

for cdata = 1:1
idata = cdata; % choose the dataset
dataf = [datadir, cell2mat(dataname(idata))];
load (dataf);
%---------------------- load data -------------------------
truelabel=truelabel{1}';
n = size(truelabel', 2);
nv = size(data, 2);
% for i=1:nv
% data{i}=data{i}';
% end
oridata_num=size(data{1},2);
K = length(unique(truelabel));
gnd = truelabel;
testing_times = 10;
options = [];
alpha_j = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100];
beta_y = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000];
lambda1 = [1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100];
options.nRepeat = 10;
options.clusteringFlag = 1;
options.maxiter=100;
options.stoperr=1e-5;

options.NeighborMode = 'KNN';
options.k = 4;
options.WeightMode = 'Binary'; % Binary  HeatKernel
miu = 1e-2;
rho = 1.2;

%---------- The missing ratios --------------------------------------------
%- There are four experiments. --------------------------------------------
%--For example-------------------------------------------------------------
%- 0 represents that all features are available.
%- 0.1 represents that 10% of features are randomly missing in each view.
%-------------------------------------------------------------------------
raitos = [0.1, 0.3, 0.5];

% initialize U{i}, P, W ...
Uv = cell(1, nv);
W=1/n*ones(n,n);
results=[];
acc = [];
nmi = [];
Pri = [];
%---------- initialize SE --------------------------------------------
SE = cell(1, nv);
miss_m1 = cell(1, nv);
for nv_idx = 1 : nv
data{nv_idx} = NormalizeFea(data{nv_idx},0);
end
  for j = 2 : 2
         alpha=alpha_j(j);
         disp(sprintf('alpha= %d ',alpha));
    for y = 6 : 6
             beta=beta_y(y);
             disp(sprintf('bata= %d ',beta));
                for g = 4 : 4
                  lambda=lambda1(g);
                  disp(sprintf('lambda= %d ',lambda));
for raito_idx = 2 : 2%length(raitos)
     disp(sprintf('È±Ê§ÂÊÎª %f ',raitos(raito_idx)));
    data_views = cell(1, nv);
    YY= cell(1, nv);
    for nv_idx = 1 : nv
     data_views{nv_idx} = data{nv_idx};
     YY{nv_idx} = data{nv_idx};
    end
    
    stream = RandStream.getGlobalStream;
    reset(stream);
    raito = 1 - raitos(raito_idx);    
    rand('state', 100);
    for nv_idx = 1 : nv
        if raito < 1
             pos = randperm(n);
             num = ceil(n * raito);%floor
             sample_pos = zeros(1, n);
             sample_pos(pos(1 : num)) = 1;
             SE{nv_idx} = diag(sample_pos);
             ind = find(sample_pos==0);
             miss_m1{nv_idx} = ind;
             SE{nv_idx}(:,ind)=[];
             YY{nv_idx}(:,ind)=0;
             data_views{nv_idx}(:,ind)=[];
        else
            SE{nv_idx} = diag(ones(1, n));
        end
    end
   
        [S, P_t, Z1, P1, Un, wv, obj] = initialize(oridata_num, data_views, Uv, YY, SE, options, K, alpha, beta, lambda, miu, rho);
        [Xv,lastbas,new_dataP, lastS,lastobj] = TCLIMC(data_views, P1, Un, Z1, S, P_t, wv, miss_m1, options, alpha, beta, lambda, miu, rho);
        clear data_views SE;

      Sum_Z = (lastS+lastS')*0.5;        
      Dd = diag(sqrt(1./(sum(Sum_Z,1)+eps)));
      An = Dd*Sum_Z*Dd;
      An(isnan(An)) = 0;
      An(isinf(An)) = 0;
try
    [Fng, ~] = eigs(An,K);
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eigs:ARPACKroutineErrorMinus14'))
        opts.tol = 1e-3;
        [Fng, ~] = eigs(An,K,opts.tol);
    else
        rethrow(ME);
    end
end
Fng(isnan(Fng))=0;
Fng = Fng./repmat(sqrt(sum(Fng.^2,2))+eps,1,K);  
        
        for iter = 1:options.nRepeat       
            pre_labels = kmeans(real(Fng),K,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
            result_cluster = ClusteringMeasure(gnd, pre_labels)*100;
            result_acc(iter)=result_cluster(1);
            result_nmi(iter)=result_cluster(2);
            result_pri(iter)=result_cluster(3);
        end
        result1=[mean(result_acc) mean(result_nmi) mean(result_pri)];
        results=[results;result1];
        fprintf('ACC: %0.2f\tNMI:%0.2f\tPri:%0.2f\n', mean(result_acc), mean(result_nmi), mean(result_pri));
        fprintf('ACCstd: %0.2f\tNMIstd:%0.2f\tPristd:%0.2f\n', std(result_acc), std(result_nmi), std(result_pri));
        disp('----------------------------------------------------------------------------------');
end
                end
   end
     clear Result;
 end

end
% X = An;
% Y = tsne(X,'Algorithm','exact','Distance','euclidean');
% set(gcf,'PaperSize',[16,11]);
% gscatter(Y(:,1), Y(:,2),gnd);
       
