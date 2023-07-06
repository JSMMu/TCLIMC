% data_e: existing data X_E
% ori_new_data: original P
% ori_bas: original U_v
% los_mark: mark of missing data
function [S, P, Z1, new_data, bas, w, obj] = initialize(data_num,data_views, Uv, YY, SE, options,K, alpha, beta, lambda, miu, rho)
% alpha ¦Â
% beta ¦Ë
maxiter = options.maxiter;
stoperr = options.stoperr;
data_e = data_views;
bas=Uv;
view_num = length(data_e);
N=size(data_e{1}, 2);
islocal = 1; % default: only update the similarities of neighbors if islocal=1
dim = zeros(1,view_num); 
sum_dim = 0;
for iv=1:view_num
    Z_ini = full(constructW(data_e{iv}',options)); 
    Z1{iv} = SE{iv}*max(Z_ini,Z_ini')*SE{iv}';
    clear Z1_ini
end
P = Z1;
for iv = 1:view_num
    C{iv} = zeros(size(Z1{iv}));
end
S0 = zeros(data_num);
for i = 1:view_num
    S0 = S0 + Z1{i};
end
S0 = S0/view_num;
for j = 1:data_num
    d_sum = sum(S0(j,:));
    if d_sum == 0
        d_sum = eps;
    end
    S0(j,:) = S0(j,:)/d_sum;
end
S = (S0+S0')/2;
D = diag(sum(S));
L = D - S;
w = ones(1,view_num)/view_num;

for view_mark = 1:view_num    
    [dim(view_mark),~] = size(data_e{view_mark});
    sum_dim = sum_dim+dim(view_mark);
end
%initialize W_v,H
sumH = 0;
for i = 1:view_num
    [ilabels,CC] = litekmeans(YY{i}', K, 'Replicates', 20);
    bas{i} = CC' + 0.1*ones(dim(i),K);  
    G = zeros(data_num,K);
    for j=1:K
        G(:,j)=(ilabels == j*ones(data_num,1));
    end 
    H{i}=G+0.1*ones(data_num,K);
    sumH = sumH + H{i};    
end
new_data = sumH'/view_num;

for it_mark = 1:maxiter  
    Z_pre = Z1;
    P_pre = P;
    temp1 = zeros(K,data_num); 
    temp2 = zeros(K,data_num); 
     for view_mark = 1:view_num      
        %%%% update wv
        SZ = S - Z1{view_num};
        distSZ = norm(SZ, 'fro')^2;
        if distSZ == 0
            distSZ = eps;
        end
        w(view_num) = 0.5/sqrt(distSZ);
        
        temp3 = data_e{view_mark}*SE{view_mark}'*new_data';%XE(v)SETPT
        temp4 = bas{view_mark}*new_data*SE{view_mark}*SE{view_mark}'*new_data'+beta*bas{view_mark};%       
        %%%% update W_v
        bas{view_mark} = bas{view_mark}.*(temp3./max(temp4,1e-10));
        %%%%%
        
        temp1 =  temp1+bas{view_mark}'*data_e{view_mark}*SE{view_mark}';
        temp2 =  temp2+bas{view_mark}'*bas{view_mark}*new_data*SE{view_mark}*SE{view_mark}';
     end

    %%%% update H 
   new_data=new_data.*((temp1 + alpha*new_data*S)./(max(temp2 + alpha*new_data*D,1e-10)));
    %%%%
    
    %%%% update Z(v)
     SumZ = 0;
    for iv = 1:view_num
        SumZ = SumZ+Z1{iv};
    end
    for iv = 1:view_num
         linshi1 = 2*lambda*YY{iv}'*YY{iv}+2*w(iv)*eye(data_num)+miu*eye(data_num);
        linshi2 = 2*lambda*YY{iv}'*YY{iv}+2*w(iv)*S+miu*P{iv}-C{iv};
        linshi = linshi1\linshi2; 
        Z = zeros(size(linshi));
        for is = 1:size(linshi,1)
           ind_c = 1:size(linshi,1);
           ind_c(is) = [];
           Z(is,ind_c) = EProjSimplex_new(linshi(is,ind_c));
        end
        Z1{iv} = Z;
    end
    clear linshi1 linshi2 linshi
    
    % ----------------- P --------------%
    Z1_tensor = cat(3, Z1{:,:});
    C_tensor = cat(3, C{:,:});
    Zv = Z1_tensor(:);
    Cv = C_tensor(:);
    [Pv, objV] = wshrinkObj(Zv + 1/miu*Cv,1/miu,[data_num,data_num,view_num],0,1);
    P_tensor = reshape(Pv, [data_num,data_num,view_num]);
    % -----------------------------------%
    for iv = 1:view_num
        P{iv} = P_tensor(:,:,iv);
        % -------- C{iv} ------%
        C{iv} = C{iv}+miu*(Z1{iv}-P{iv});
    end
    clear Z1_tensor C_tensor Zv Cv Pv P_tensor
   
    % update S
    dist = L2_distance_1(new_data,new_data);
    S = zeros(data_num);
    for i=1:data_num
        idx = zeros();
        for v = 1:view_num
            s0 = Z1{v}(i,:);
            idx = [idx,find(s0>0)];
        end
        idxs = unique(idx(2:end));
        if islocal == 1
            idxs0 = idxs;
        else
            idxs0 = 1:data_num;
        end
        for v = 1:view_num
            s1 = Z1{v}(i,:);
            si = s1(idxs0);
            di = dist(i,idxs0);
            mw = view_num*w(v);
            lmw = alpha/mw;
            q(v,:) = si-0.5*lmw*di;
        end
        S(i,idxs0) = SloutionForP20(q,view_num);
        clear q;
    end
    S = (S+S')/2;
    D = diag(sum(S));
    L = D - S;
    
    %%%%
    miu = min(miu*rho, 1e10);
    ever_Z = 0;
    ever_P = 0;
    leqm1 = 0;
    leqm2 = 0;
    leqm3 = 0;
    %% check convergence
    for iv = 1:view_num
        Rec_error1 = data_e{iv}-bas{iv}*new_data*SE{iv};
        Rec_error2 = YY{iv}-YY{iv}*Z1{iv};
        Rec_error3 = S-SumZ;
        leqm1 = max(leqm1,max(abs(Rec_error1(:))));
        leqm2 = max(leqm2,max(abs(Rec_error2(:))));
        leqm3 = max(leqm3,max(abs(Rec_error3(:))));
        leq{iv} = Z1{iv}-P{iv};
        ever_Z = max(ever_Z,max(abs(Z1{iv}(:)-Z_pre{iv}(:))));
        ever_P = max(ever_P,max(abs(P{iv}(:)-P_pre{iv}(:))));       
    end
    leqm = cat(3, leq{:,:});
    leqms = max(abs(leqm(:)));
    clear leq leqm Rec_error1 Rec_error2 Rec_error3
    err = min([leqm1,leqm2,leqm3,leqms]);
    %err = leqms;
    fprintf('iter = %d, miu = %.3f, everZ = %.d, everP = %.d, err = %.8f\n'...
            , it_mark,miu,ever_Z,ever_P,err);
    obj(it_mark) = err;   
    if err < 1e-6
        it_mark
        break;
    end
end
   norm_new_data = repmat(sqrt(sum(new_data.*new_data)),size(new_data,1),1);
   norm_new_data = max(norm_new_data,1e-10);
   new_data = new_data./norm_new_data;