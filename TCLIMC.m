% data_e: existing data X_E
% ori_new_data: original P
% ori_bas: original U_v
% los_mark: mark of missing data
function [data,bas,new_data, S, obj] = TCLIMC(data_e,ori_new_data,ori_bas, Z, S, P, wv, los_mark,options, alpha, beta, lambda, miu, rho)
% alpha ¦Â
% beta ¦Ë
maxiter = options.maxiter;
stoperr = options.stoperr;
view_num = length(data_e);
[new_dim,data_num] = size(ori_new_data);
dim = zeros(1,view_num); 
tq_l = cell(1,view_num);   
kz_l = cell(1,view_num); 
kz_e = cell(1,view_num); 
tot_mark = 1:data_num;   
data_l = cell(1,view_num); 
data = cell(1,view_num);  
islocal = 1;
sum_dim = 0;
for view_mark = 1:view_num    
    [dim(view_mark),~] = size(data_e{view_mark});
    sum_dim = sum_dim+dim(view_mark);
     los_mark{view_mark} = sort(los_mark{view_mark});  
     ext_mark = setdiff(tot_mark,los_mark{view_mark});
     tq_l{view_mark} = eye(data_num);
     tq_l{view_mark}(:,ext_mark) = [];
     l_num = length(los_mark{view_mark}); 
     e_num = data_num - l_num;  
     kz_l{view_mark} = zeros(l_num,data_num);
     kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
     kz_e{view_mark} = zeros(e_num,data_num);
     kz_e{view_mark}(:,ext_mark) = eye(e_num);
end 

m =zeros(1,view_num);
I = cell(1,view_num);
for i=1:view_num
    m(i)=size(ori_bas{i},1);
    I{i}=ones(m(i),data_num);
end
for i=1:1
    I1=eye(new_dim,new_dim);
    I2=eye(data_num);
end

new_data = ori_new_data;  
bas = ori_bas;
D = diag(sum(S));
L = D - S;
for iv = 1:view_num
    C{iv} = zeros(size(Z{iv}));
end

for it_mark = 1:maxiter 
    Z_pre = Z;
    P_pre = P;
    temp1 = 0; 
    temp2 = 0; 
     for view_mark = 1:view_num
        %%%% update X_v
         data_l{view_mark} =((bas{view_mark}*new_data)/(I2+lambda*I2-lambda*Z{view_mark}-lambda*Z{view_mark}'+lambda*Z{view_mark}*Z{view_mark}'))*tq_l{view_mark}; 
        data{view_mark} = data_l{view_mark}*kz_l{view_mark} + data_e{view_mark}*kz_e{view_mark}; 
        %%%%
        
         %%%% update wv
        SZ = S - Z{view_num};
        distSZ = norm(SZ, 'fro')^2;
        if distSZ == 0
            distSZ = eps;
        end
        wv(view_num) = 0.5/sqrt(distSZ);
 
        temp3 = data{view_mark}*new_data';
        temp4 = bas{view_mark}*new_data*new_data'+beta*bas{view_mark};
        
        %%%% update W_v
        bas{view_mark} = bas{view_mark}.*(temp3./max(temp4,1e-10));
        %%%%%

        temp1 =  temp1+bas{view_mark}'*data{view_mark};
        temp2 =  temp2+bas{view_mark}'*bas{view_mark}*new_data;
     end
     
    %%%% update H
     new_data=new_data.*((temp1 + alpha*new_data*S)./(max(temp2 + alpha*new_data*D,1e-10)));
    %%%%
    
    %%%% update Z(v)
     SumZ = 0;
    for iv = 1:view_num
        SumZ = SumZ+Z{iv};
    end
    for iv = 1:view_num
        linshi1 = 2*lambda*data{iv}'*data{iv}+2*wv(iv)*eye(data_num)+miu*eye(data_num);
        linshi2 = 2*lambda*data{iv}'*data{iv}+2*wv(iv)*S+miu*P{iv}-C{iv};
        linshi = linshi1\linshi2;
        Z0 = zeros(size(linshi));
        for is = 1:size(linshi,1)
           ind_c = 1:size(linshi,1);
           ind_c(is) = [];
           Z0(is,ind_c) = EProjSimplex_new(linshi(is,ind_c));
        end
        Z{iv} = Z0;
    end
    clear linshi1 linshi2 linshi

    % ----------------- P --------------%
    Z1_tensor = cat(3, Z{:,:});
    C_tensor = cat(3, C{:,:});
    Zv = Z1_tensor(:);
    Cv = C_tensor(:);
    [Pv, objV] = wshrinkObj(Zv + 1/miu*Cv,1/miu,[data_num,data_num,view_num],0,1);
    P_tensor = reshape(Pv, [data_num,data_num,view_num]);
    % -----------------------------------%
    
    for iv = 1:view_num
        P{iv} = P_tensor(:,:,iv);
        % -------- C{iv} ------%
        C{iv} = C{iv}+miu*(Z{iv}-P{iv});
    end
     clear Z1_tensor C_tensor Zv Cv Pv P_tensor
    
     % update S
    dist = L2_distance_1(new_data,new_data);
    S = zeros(data_num);
    for i=1:data_num
        idx = zeros();
        for v = 1:view_num
            s0 = Z{v}(i,:);
            idx = [idx,find(s0>0)];
        end
        idxs = unique(idx(2:end));
        if islocal == 1
            idxs0 = idxs;
        else
            idxs0 = 1:data_num;
        end
        for v = 1:view_num
            s1 = Z{v}(i,:);
            si = s1(idxs0);
            di = dist(i,idxs0);
            mw = view_num*wv(v);
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
        Rec_error1 = data{iv}-bas{iv}*new_data;
        Rec_error2 = data{iv}-data{iv}*Z{iv};
        Rec_error3 = S-SumZ;
        leqm1 = max(leqm1,max(abs(Rec_error1(:))));
        leqm2 = max(leqm2,max(abs(Rec_error2(:))));
        leqm3 = max(leqm3,max(abs(Rec_error3(:))));
        leq{iv} = Z{iv}-P{iv};
        ever_Z = max(ever_Z,max(abs(Z{iv}(:)-Z_pre{iv}(:))));
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
