function V = UpdateV_initialize(X,SE,U,V,S,D,L,alpha,viewNum)

time = 0;
f = 0;
while 1
    time = time +1;
    sumVUUminus = 0;
    sumVUUplus = 0;
    sumXUminus = 0;
    sumXUplus = 0;
    for i = 1:viewNum
        XU = U{i}'*X{i};
        absXU = abs(XU);
        XUplus = (absXU + XU)/2;
        XUminus = (absXU - XU)/2;
        
        UU = U{i}'*U{i};
        absUU = abs(UU);
        UUplus = (absUU + UU)/2;
        UUminus = (absUU - UU)/2;
            
        sumXUminus = sumXUminus + XUminus*SE{i}';
        sumXUplus = sumXUplus + XUplus*SE{i}';
        
        sumVUUplus = sumVUUplus + UUplus*V*SE{i}*SE{i}';
        sumVUUminus = sumVUUminus + UUminus*V*SE{i}*SE{i}';
    end

    V = V.*sqrt((sumXUplus + alpha*V*S + sumVUUminus + alpha*V*D)./(max(sumXUminus + alpha*V*S + sumVUUplus + alpha*V*D,1e-10)));  
    
    ff = 0;
    for i = 1:viewNum
        tmp = (X{i} - U{i}*V*SE{i});
        ff = ff + sum(sum(tmp.^2))+alpha*trace(V*L*V');
    end
    if abs((ff-f)/f)<1e-4 | abs(ff-f)>1e100 | time == 30	
        break;
    end
    f = ff; 
end
end


