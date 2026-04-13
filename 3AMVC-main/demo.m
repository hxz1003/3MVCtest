

clear all;
warning off;
addpath(genpath('./'));

%% dataset
ds={'MFeat_2Views'};
dsPath = '...\3AMVC\dataset\';
resultdir = '.\res\';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};



for dsi =1:length(ds)
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
k = length(unique(Y)) ;
n = size(Y,1);
v = length(X);
beta = 100;
lambda = 10^4;

    %%
    for id =  1:length(beta)
        for ic = 1:length(lambda)
            for it = 1 : 1
                for iv = 1:v
                    tic
                    [res_neighbor,time_neighbor,label_neighbor,object,theta,k_neighbor] = Neighbor(X{iv},Y);
                    thetaall{iv,:} = theta;
                    object_sum(iv,:) = sum(object);
                end
                [~,target_view] = min(object_sum);
                [U,A,Z,iter,obj] = algo_qp(X,Y,thetaall,beta(id),lambda(ic),target_view); % X,Y,lambda,d,theta
                [result(it,:),~] = myNMIACCwithmean(U,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
                times(it)  = toc;
            end
            max_id = find(result(:,1)==max(result(:,1)));
            resmax = result(max_id,:);
            timem = mean(times);
            fprintf('Beta:%d\t Lambda:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[beta(id) lambda(ic) resmax(1) resmax(2) resmax(3) resmax(4) timem]);
        end
    end

end


