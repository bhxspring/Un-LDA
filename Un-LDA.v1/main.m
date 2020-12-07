clc
clear
close all
addpath('..\data');
addpath('..\publicCode');
rng(123);
datasets = {'cars'};
d1 = 'results_kmeans';
d2 = 'results_LDA-Km';
d3 = 'results_unsupervisedR-RatioTraceLDA';
d4 = 'results_unsupervisedTraceRatioLDA';
d5 = 'results_R-RTLDA+Kmeans';
d6 = 'results_TRLDA+Kmeans';
if ~exist(d1,'dir')
    mkdir(d1);
end
if ~exist(d2,'dir')
    mkdir(d2);
end
if ~exist(d3,'dir')
    mkdir(d3);
end
if ~exist(d4,'dir')
    mkdir(d4);
end
if ~exist(d5,'dir')
    mkdir(d5);
end
if ~exist(d6,'dir')
    mkdir(d6);
end

times = 100;
maxIter = 100;
for s = 1:length(datasets)
    name = datasets{s};
    load([name,'.mat']);
    numClass = length(unique(Y));
    [nSamp,nDim] = size(X);
    H = eye(nSamp)-ones(nSamp,nSamp)/nSamp;
    St = X'*H*X;  

    W0 = pca(H*X);
    StartInd = zeros(nSamp,times);
    for i=1:times
        StartInd(:,i) = randsrc(nSamp,1,1:numClass);
    end
    
    %%%%%%%%%%%%%%%%%%%%%% kmeans %%%%%%%%%%%%%%%%%%%%%%%
    [acc,NMI,purity] = deal(zeros(times,1));
    parfor rep=1:times
        [Ypre,~,~,~] = kmeans_ldj(H*X,StartInd(:,rep));
        result = ClusteringMeasure(Y, Ypre);
        acc(rep) = result(1);
        NMI(rep) = result(2);
        purity(rep) = result(3);
    end
    mean_acc1 = mean(acc);mean_NMI1 = mean(NMI);mean_purity1 = mean(purity);
    std_acc1 = std(acc);std_NMI1 = std(NMI);std_purity1 = std(purity);
    max_acc1 = max(acc);max_NMI1 = max(NMI);max_purity1 = max(purity);
    save([d1,'\',name,'_times=',num2str(times),'.mat'],'mean_acc1','std_acc1','max_acc1','mean_NMI1','std_NMI1','max_NMI1','mean_purity1','std_purity1','max_purity1');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gamma = 10.^(-4:4);
    numReducedDim = min(numClass-1,nDim);
    W = W0(:,1:numReducedDim);
    nGamma = length(gamma);
    [obj1,obj2] = deal(zeros(maxIter,nGamma));
    [mean_acc2,mean_NMI2,mean_purity2,std_acc2,std_NMI2,std_purity2,max_acc2,max_NMI2,max_purity2,...
    mean_acc3,mean_NMI3,mean_purity3,std_acc3,std_NMI3,std_purity3,max_acc3,max_NMI3,max_purity3] = deal(zeros(nGamma,1));
    for i=1:nGamma        
        %%%%%%%%%%%%%%%%%%%%%% LDA-Km %%%%%%%%%%%%%%%%%%%%%%%
        [acc,NMI,purity] = deal(zeros(times,1));
        for rep=1:times
            obj_1 = zeros(maxIter,1);
            it = 0;
            W1 = W;
            obj_old = 0;
            obj_new = NaN;
            Ypre = StartInd(:,rep);
            while ~isequal(obj_old,obj_new)&&it<maxIter
                it = it+1;
                obj_old = obj_new;
                
                M = (W1'*X'*H)';
                [Ypre,~,center,obj] = kmeans_ldj(M,Ypre);
                
                Yp = full(ind2vec(Ypre',numClass))';
                Sb = X'*H*Yp*(Yp'*Yp)^-1*Yp'*H'*X;
                Sw = St-Sb;
                Sw = Sw+gamma(i)*eye(size(Sw));
                
                model = GEVD(Sb,Sw);
                W1 = model.W(:,end-numReducedDim+1:end);
                
                obj_new = trace(W1'*St*W1)/trace(W1'*Sw*W1)-1;
                obj_1(it) = obj_new;

            end
            if it == maxIter
                disp(['Warnning: the LDA-Km does not converge within ',num2str(maxIter),' iterations!','(',name,')']);
            end
            result = ClusteringMeasure(Y, Ypre);
            acc(rep) = result(1);
            NMI(rep) = result(2);
            purity(rep) = result(3);
        end        
        mean_acc2(i) = mean(acc);mean_NMI2(i) = mean(NMI);mean_purity2(i) = mean(purity);
        std_acc2(i) = std(acc);std_NMI2(i) = std(NMI);std_purity2(i) = std(purity);
        max_acc2(i) = max(acc);max_NMI2(i) = max(NMI);max_purity2(i) = max(purity);  
        obj1(:,i) = obj_1;
        %%%%%%%%%%%%% unsupervisedR-RatioTraceLDA %%%%%%%%%%%%%
        [acc,NMI,purity] = deal(zeros(times,1));
        for rep=1:times
            obj_2 = zeros(maxIter,1);
            Stt = St+gamma(i)*eye(size(St,1));
            it = 0;
            W2 = W;
            obj_old = 0;
            obj_new = NaN;
            Ypre = StartInd(:,rep);
            while ~isequal(obj_old,obj_new)&&it<maxIter
                it = it+1;
                obj_old = obj_new;
                
                M = ((W2'*Stt*W2)^(-0.5)*W2'*X'*H)';
                [Ypre,~,~,obj] = kmeans_ldj(M,Ypre);
                
                Yp = full(ind2vec(Ypre',numClass))';
                Sb = X'*H*Yp*(Yp'*Yp)^-1*Yp'*H'*X;
                
                model = GEVD(Sb,Stt);
                W2 = model.W(:,end-numReducedDim+1:end);
                
                obj_new = trace((W2'*Stt*W2)^-1*W2'*Sb*W2);
                obj_2(it) = obj_new;

            end
            if it == maxIter
                disp(['Warnning: the unsupervisedR-RatioTraceLDA does not converge within ',num2str(maxIter),' iterations!','(',name,')']);
            end
            result = ClusteringMeasure(Y, Ypre);
            acc(rep) = result(1);
            NMI(rep) = result(2);
            purity(rep) = result(3);
        end        
        mean_acc3(i) = mean(acc);mean_NMI3(i) = mean(NMI);mean_purity3(i) = mean(purity);
        std_acc3(i) = std(acc);std_NMI3(i) = std(NMI);std_purity3(i) = std(purity);
        max_acc3(i) = max(acc);max_NMI3(i) = max(NMI);max_purity3(i) = max(purity);  
        obj2(:,i) = obj_2;
    end
    save([d2,'\',name,'_times=',num2str(times),'.mat'],'obj1','mean_acc2','std_acc2','max_acc2','mean_NMI2','std_NMI2','max_NMI2','mean_purity2','std_purity2','max_purity2');
    save([d3,'\',name,'_times=',num2str(times),'.mat'],'obj2','mean_acc3','std_acc3','max_acc3','mean_NMI3','std_NMI3','max_NMI3','mean_purity3','std_purity3','max_purity3');
    %%%%%%%%%%%%% unsupervisedTraceRatioLDA %%%%%%%%%%%%%%
    m = size(W0,2);
    obj3 = zeros(maxIter,m);
    [mean_acc4,mean_NMI4,mean_purity4,std_acc4,std_NMI4,std_purity4,max_acc4,max_NMI4,max_purity4] = deal(zeros(m,1));
    for i=1:m
        [acc,NMI,purity] = deal(zeros(times,1));
        for rep=1:times
            obj_3 = zeros(maxIter,1);
            W_TraceRatio = W0(:,1:i);
            it = 0;
            W3 = W_TraceRatio;
            obj_old = 0;
            obj_new = NaN;
            Ypre = StartInd(:,rep);
            while ~isequal(obj_old,obj_new)&&it<maxIter
                it = it+1;
                obj_old = obj_new;
                
                M = (W3'*X'*H)';
                [Ypre,~,~,obj] = kmeans_ldj(M,Ypre);
                
                Yp = full(ind2vec(Ypre',numClass))';
                Sb = X'*H*Yp*(Yp'*Yp)^-1*Yp'*H'*X;
                
                [W3, ~] = TraceRatio(Sb,St,i);
                obj_new = trace(W3'*Sb*W3)/trace(W3'*St*W3);
                obj_3(it) = obj_new;
                
            end
            if it == maxIter
                disp(['Warnning: the unsupervisedTraceRatioLDA does not converge within ',num2str(maxIter),' iterations!','(',name,')']);
            end
            result = ClusteringMeasure(Y, Ypre);
            acc(rep) = result(1);
            NMI(rep) = result(2);
            purity(rep) = result(3);  
        end
        mean_acc4(i) = mean(acc);mean_NMI4(i) = mean(NMI);mean_purity4(i) = mean(purity);
        std_acc4(i) = std(acc);std_NMI4(i) = std(NMI);std_purity4(i) = std(purity);
        max_acc4(i) = max(acc);max_NMI4(i) = max(NMI);max_purity4(i) = max(purity);   
        obj3(:,i) = obj_3;
    end
    save([d4,'\',name,'_times=',num2str(times),'.mat'],'obj3','mean_acc4','std_acc4','max_acc4','mean_NMI4','std_NMI4','max_NMI4','mean_purity4','std_purity4','max_purity4');

    %%%%%%%%%%%%% R-RTLDA+Kmeans %%%%%%%%%%%%%
    [mean_acc5,mean_NMI5,mean_purity5,std_acc5,std_NMI5,std_purity5,max_acc5,max_NMI5,max_purity5] = deal(zeros(nGamma,1)); 
    for i=1:nGamma          
        [acc,NMI,purity] = deal(zeros(times,1));
        parfor rep=1:times
            Yp = full(ind2vec(Y',numClass))';
            Sb = X'*H*Yp*(Yp'*Yp)^-1*Yp'*H'*X;
            Sw = St-Sb;
            Sw = Sw+gamma(i)*eye(size(Sw));
            model = GEVD(Sb,Sw);
            W5 = model.W(:,end-numReducedDim+1:end);
            
            [Ypre,~,~,~] = kmeans_ldj(H*X*W5,StartInd(:,rep));
            result = ClusteringMeasure(Y, Ypre);
            acc(rep) = result(1);
            NMI(rep) = result(2);
            purity(rep) = result(3);
        end
        mean_acc5(i) = mean(acc);mean_NMI5(i) = mean(NMI);mean_purity5(i) = mean(purity);
        std_acc5(i) = std(acc);std_NMI5(i) = std(NMI);std_purity5(i) = std(purity);
        max_acc5(i) = max(acc);max_NMI5(i) = max(NMI);max_purity5(i) = max(purity);
    end
    save([d5,'\',name,'_times=',num2str(times),'.mat'],'mean_acc5','std_acc5','max_acc5','mean_NMI5','std_NMI5','max_NMI5','mean_purity5','std_purity5','max_purity5');

    %%%%%%%%%%%%% TRLDA+Kmeans %%%%%%%%%%%%%
    [mean_acc6,mean_NMI6,mean_purity6,std_acc6,std_NMI6,std_purity6,max_acc6,max_NMI6,max_purity6] =deal(zeros(m,1));
    for i=1:m
        [acc,NMI,purity] = deal(zeros(times,1));
        parfor rep=1:times
            Yp = full(ind2vec(Y',numClass))';
            Sb = X'*H*Yp*(Yp'*Yp)^-1*Yp'*H'*X;
            [W6, ~] = TraceRatio(Sb,St,i);
            
            [Ypre,~,~,~] = kmeans_ldj(H*X*W6,StartInd(:,rep));
            result = ClusteringMeasure(Y, Ypre);
            acc(rep) = result(1);
            NMI(rep) = result(2);
            purity(rep) = result(3);
        end
        mean_acc6(i) = mean(acc);mean_NMI6(i) = mean(NMI);mean_purity6(i) = mean(purity);
        std_acc6(i) = std(acc);std_NMI6(i) = std(NMI);std_purity6(i) = std(purity);
        max_acc6(i) = max(acc);max_NMI6(i) = max(NMI);max_purity6(i) = max(purity);
    end
    save([d6,'\',name,'_times=',num2str(times),'.mat'],'mean_acc6','std_acc6','max_acc6','mean_NMI6','std_NMI6','max_NMI6','mean_purity6','std_purity6','max_purity6');

end


