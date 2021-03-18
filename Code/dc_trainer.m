function [U,S,V,threshold,w,sort1,sort2] = dc_trainer(data1,dta2,feature)
    
    n1 = size(data1,2);
    n2 = size(dta2,2);
    [U,S,V] = svd([data1 dta2],'econ'); 
    projection = S*V';
    U = U(:,1:feature); % Add this in
    data1 = projection(1:feature,1:n1);
    data2 = projection(1:feature,n1+1:n1+n2);
    m1 = mean(data1,2);
    m2 = mean(data2,2);

    Sw = 0;
    for k=1:n1
        Sw = Sw + (data1(:,k)-m1)*(data1(:,k)-m1)';
    end
    for k=1:n2
        Sw = Sw + (data2(:,k)-m2)*(data2(:,k)-m2)';
    end
    Sb = (m1-m2)*(m1-m2)';
    
    [V2,D] = eig(Sb,Sw);
    [lambda,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    v1 = w'*data1;
    v2 = w'*data2;
    
    if mean(v1)>mean(v2)
        w = -w;
        v1 = -v1;
        v2 = -v2;
    end
    
    % Don't need plotting here
    sort1 = sort(v1);
    sort2 = sort(v2);
    t1 = length(sort1);
    t2 = 1;
    while sort1(t1)>sort2(t2)
    t1 = t1-1;
    t2 = t2+1;
    end
    threshold = (sort1(t1)+sort2(t2))/2;

    % We don't need to plot results
end

