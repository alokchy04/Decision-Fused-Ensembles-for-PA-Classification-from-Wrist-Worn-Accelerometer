function [stat] = mainfile()
    clear; clc;
    rng(1024); % for reproducibility
    
    %data file collect - START        
    %dataset 1
    path{1} = strcat(pwd, '\FET\');
    file{1}='hand-DS1.csv'; 
    opt{1} = 5;    
    %dataset 2
    path{2} = strcat(pwd, '\FET\');
    file{2}='hand-DS2.csv'; 
    opt{2} = 4; 
    %dataset 3
    path{3} = strcat(pwd, '\FET\');
    file{3}='hand-DS3.csv'; 
    opt{3} = 3;
    %data file collect - END
    
    
    o=1;
    while(o < 4)        
        clearvars -except o path file opt;
        
        %set up params
        classification_methods = {'RandomForest' 'Bagged' 'Adaboost' 'BDT' 'KNN' 'SVM' 'NN'};   
        isinFusion = {'0' '0' '0' '1' '1' '1' '1'};        
        fusion = 1; leaveout = 1; fold = 0; split = 0;
 
        fileName=file{o};
        pathName=path{o};
        disp(fileName);  
        
        %read feature file - START
        [fetData, fetHeader, error] = read_CSV_File(strcat(pathName,fileName));        
        label=fetData(:,end);  
        idnty=fetData(:,end-1);
        fetData=fetData(:,1:end-2);      
        %read feature file - END
        
        %incase any NaN rows, remove - START   
        if(leaveout == 1)
            total = [fetData idnty label];
            nanIndicator = ~any(isnan(total),2);
            total = total(nanIndicator,:);
            fetData = total(:,1:size(total,2)-size(label,2)-1);
            idnty = total(:,end-1);
            label = total(:,size(total,2)-size(label,2)+1:size(total,2));
        end
        %incase any NaN rows, remove - END         
        data = fetData;
                
        %Classification - START                                                                                                                                                                                            
        newlabel_Adaboost= zeros(size(label,1),size(label,2));
        newlabel_KNN= zeros(size(label,1),size(label,2));    
        newlabel_SVM= zeros(size(label,1),size(label,2));
        newlabel_DNN= zeros(size(label,1),size(label,2));
        newlabel_NN= zeros(size(label,1),size(label,2));
        newlabel_BDT= zeros(size(label,1),size(label,2));   
        newlabel_WMV= zeros(size(label,1),size(label,2));
        newlabel_NB= zeros(size(label,1),size(label,2));
        newlabel_BKS= zeros(size(label,1),size(label,2));
        no_of_class = length(unique(label));    
        tempStats_Adaboost=[];
        tempStats_KNN=[];
        tempStats_SVM=[];
        tempStats_RandomForest=[];
        tempStats_Bagged=[];
        tempStats_DNN=[];
        tempStats_NN=[];
        tempStats_BDT=[];
        tempStats_WMV=[];
        tempStats_NB=[];
        tempStats_BKS=[];

        loop = 0;
        if(leaveout == 1)
            indices = idnty;
            loop = max(idnty);
        end
        stat.nooffold = loop;
        stat.split = split;
        
        
        testIndex2 = [];
        for k = 1:1:loop
            fprintf('Fold %d...\n',k);  
            
            if(fold ~= 0 || leaveout == 1) %folding/leaveout
                testIndex = (indices == k); trainIndex = ~testIndex;                
                testIndex1 = find(indices==k);
            end
            testIndex = logical(testIndex);
            trainIndex = logical(trainIndex);

            if sum(testIndex)==0
                fprintf('continue\n'); 
                continue;            
            end

            trainData = data(trainIndex,:);
            testData = data(testIndex,:);
            trainLabel = label(trainIndex,:);
            testLabel = label(testIndex,:);
            trainDataDNN = data(trainIndex,:);
            testDataDNN = data(testIndex,:);
            trainLabelDNN = label(trainIndex,:);
            testLabelDNN = label(testIndex,:);            
            
            % Normalise - START
            fs_mean = zeros(1,size(trainData,2));
            fs_std = zeros(1,size(trainData,2));
            for m=1:1:size(trainData,2)
                fs_mean(1,m)=mean(trainData(:,m));
                fs_std(1,m)=std(trainData(:,m));
                for n=1:1:size(trainData(:,m),1)
                    trainData(n,m)=(trainData(n,m)-fs_mean(1,m))/fs_std(1,m);
                end
            end
            
            for m=1:1:size(testData,2)
                for n=1:1:size(testData(:,m),1)
                    testData(n,m)=(testData(n,m)-fs_mean(1,m))/fs_std(1,m);
                end
            end
            % Normalise - END
            
            %Feature Selection - START 
            [trainData, dataHeader, trainLabel, minCorr, numoffeatures] = correlation_based_FS(trainData, fetHeader, trainLabel, 0, 0.25);                                                                                                  %[data, dataHeader, label, minCorr] = sequential_FS(fetData, fetHeader, label, 30);    
            %csvwrite_with_headers(strcat(pathName,['temp_All_0.25_' num2str(k) '_' fileName]), [trainData trainLabel], [dataHeader fetHeader{end}]);
            temp=[];
            for m=1:1:numoffeatures
                index = strfind(fetHeader,dataHeader{m});
                index = find(not(cellfun('isempty', index)));
                                
                if(~isempty(index))
                    temp=[temp testData(:, index)];
                end
            end
            testData = temp;
            
            %for DNN - no normalization 
            [trainDataDNN, dataHeader, trainLabelDNN, minCorr, numoffeatures] = correlation_based_FS(trainDataDNN, fetHeader, trainLabelDNN, 0, 0.25);                                                                                                   %[data, dataHeader, label, minCorr] = sequential_FS(fetData, fetHeader, label, 30);    
            %csvwrite_with_headers(strcat(pathName,['temp_DNN_0.25_' num2str(k) '_' fileName]), [trainDataDNN trainLabelDNN], [dataHeader fetHeader{end}]);
            temp=[];
            for m=1:1:numoffeatures
                index = strfind(fetHeader,dataHeader{m});
                index = find(not(cellfun('isempty', index)));
                                
                if(~isempty(index))
                    temp=[temp testDataDNN(:, index)];
                end
            end
            testDataDNN = temp;
            %Feature Selection - END 
            
            for j = 1:1:size(classification_methods,2)
                %ADABOOST
                if(strcmpi(classification_methods{j},'Adaboost'))
                    learners = 'Discriminant'; %|| 'Tree' - REQUIRED FOR ADABOOST                    
                    if(no_of_class > 2)
                        model = fitensemble(trainData, trainLabel, 'AdaboostM2', 100, learners);
                    else
                        model = fitensemble(trainData, trainLabel, 'AdaboostM1', 100, learners);                
                    end
                    prediction_Adaboost = predict(model, testData);
                    trPred_Adaboost = predict(model, trainData);

                    %t = confusionmatStats(trainLabel, trPred_Adaboost);
                    %fprintf('Adaboost-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_Adaboost);
                    %fprintf('Adaboost-testAccuracy - %f\n', t.accuracy);
                end
                %KNN
                if(strcmpi(classification_methods{j},'KNN'))
                    model = fitcknn(...
                        trainData, ...
                        trainLabel, ...                        
                        'NumNeighbors', 7, ...
                        'Standardize',1);
                    prediction_KNN = predict(model, testData);
                    trPred_KNN = predict(model, trainData);

                    %t = confusionmatStats(trainLabel, trPred_KNN);
                    %fprintf('KNN-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_KNN);
                    %fprintf('KNN-testAccuracy - %f\n', t.accuracy);
                end
                %SVM
                if(strcmpi(classification_methods{j},'SVM'))
                    %t = templateSVM('Standardize',1,'KernelFunction','gaussian'); 
                    t = templateSVM('KernelFunction','linear');
                    model = fitcecoc(trainData,trainLabel,'Learners',t);
                    prediction_SVM = predict(model, testData);
                    trPred_SVM = predict(model, trainData);

                    %t = confusionmatStats(trainLabel, trPred_SVM);
                    %fprintf('SVM-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_SVM);
                    %fprintf('SVM-testAccuracy - %f\n', t.accuracy);
                end
                %Random Forest
                if(strcmpi(classification_methods{j},'RandomForest'))
                    nTrees = 20;
                    model = TreeBagger(nTrees,trainData,trainLabel, 'Method', 'classification');
                    prediction_RandomForest = str2double(predict(model, testData));
                    trPred_RandomForest = str2double(predict(model, trainData));

                    %t = confusionmatStats(trainLabel, trPred_RandomForest);
                    %fprintf('RandomForest-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_RandomForest);
                    %fprintf('RandomForest-testAccuracy - %f\n', t.accuracy);
                end
                %Bagged
                if(strcmpi(classification_methods{j},'Bagged'))
                    nTrees = 20;
                    model = TreeBagger(nTrees,trainData,trainLabel, 'Method', 'classification', 'NumPredictorsToSample', 'all');
                    prediction_Bagged = str2double(predict(model, testData));
                    trPred_Bagged = str2double(predict(model, trainData));

                    %t = confusionmatStats(trainLabel, trPred_Bagged);
                    %fprintf('Bagged-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_Bagged);
                    %fprintf('Bagged-testAccuracy - %f\n', t.accuracy);
                end
                %DNN
                if(strcmpi(classification_methods{j},'DNN'))
                    level = 2;
                    featureCount = [35; 20]; 

                    % --Prepare-inputs-Start
                    newtrainData = trainDataDNN';            
                    newtestData = testDataDNN';

                    temp = trainLabelDNN';
                    [c, ia, ic] = unique(temp,'sorted');
                    newtrainLabel = zeros(size(c,2), size(trainLabelDNN,1));
                    for i=1:1:size(c,2)
                        index = find(temp==c(1,i));
                        newtrainLabel(i,index)=1;
                    end
                    % --Prepare-inputs-End

                    tempFeatures = newtrainData;
                    for i=1:1:level
                        hiddenSize = featureCount(i,1);
                        autoenc1 = trainAutoencoder(tempFeatures,hiddenSize,...
                            'L2WeightRegularization',0.001,...
                            'SparsityRegularization',4,...
                            'SparsityProportion',0.05,...
                            'MaxEpochs',1000,...
                            'DecoderTransferFunction','purelin');                
                        tempFeatures = encode(autoenc1,tempFeatures); 
                        % view(autoenc1);

                        if(i==1)
                            deepnet = autoenc1;
                        else 
                            deepnet = stack(deepnet, autoenc1);
                        end
                    end
                    softnet = trainSoftmaxLayer(tempFeatures,newtrainLabel,'LossFunction','crossentropy',...
                            'MaxEpochs',1000);
                    deepnet = stack(deepnet, softnet);
 
                    deepnet = train(deepnet,newtrainData,newtrainLabel);
                    tempprediction = deepnet(newtestData);
                    temptrPred = deepnet(newtrainData);

                    % --Prepare-results-Start
                    temp = vec2ind(tempprediction);
                    prediction_DNN=zeros(size(testData,1),1);
                    for i=1:1:size(c,2)
                        index = find(temp==i);
                        prediction_DNN(index,1)=c(1,i);
                    end

                    temp = vec2ind(temptrPred);
                    trPred_DNN=zeros(size(testData,1),1);
                    for i=1:1:size(c,2)
                        index = find(temp==i);
                        trPred_DNN(index,1)=c(1,i);
                    end
                    % --Prepare-results-End

                    %t = confusionmatStats(trainLabel, trPred_DNN);
                    %fprintf('DNN-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_DNN);
                    %fprintf('DNN-testAccuracy - %f\n', t.accuracy);
                end
                %NN
                if(strcmpi(classification_methods{j},'NN'))
                    
                    % --Prepare-inputs-Start
                    newtrainData = trainData';            
                    newtestData = testData';

                    temp = trainLabel';
                    [c, ia, ic] = unique(temp,'sorted');
                    newtrainLabel = zeros(size(c,2), size(trainLabel,1));
                    for i=1:1:size(c,2)
                        index = find(temp==c(1,i));
                        newtrainLabel(i,index)=1;
                    end
                    % --Prepare-inputs-End
                    
                    setdemorandstream(391418381);
                    net = patternnet(50);
                    net.trainParam.epochs= 250;
                    net.trainParam.goal=0;
                    net.trainParam.lr=0.001;
                    net.performFcn = 'sse';
                    
                    [net,tr] = train(net,newtrainData,newtrainLabel);                    
                    % view(net);
                    % nntraintool;
                    % plotperform(tr);
                    tempprediction = net(newtestData);                    
                    temptrPred = net(newtrainData);
                    
                    % --Prepare-results-Start
                    temp = vec2ind(tempprediction);
                    prediction_NN=zeros(size(testData,1),1);
                    for i=1:1:size(c,2)
                        index = find(temp==i);
                        prediction_NN(index,1)=c(1,i);
                    end

                    temp = vec2ind(temptrPred);
                    trPred_NN=zeros(size(testData,1),1);
                    for i=1:1:size(c,2)
                        index = find(temp==i);
                        trPred_NN(index,1)=c(1,i);
                    end
                    % --Prepare-results-End

                    %t = confusionmatStats(trainLabel, trPred_NN);
                    %fprintf('NN-trainAccuracy - %f\n', t.accuracy);
                    %t = confusionmatStats(testLabel, prediction_NN);
                    %fprintf('NN-testAccuracy - %f\n', t.accuracy);
                end
                %BDT
                if(strcmpi(classification_methods{j},'BDT'))
                    model = fitctree(...
                        trainData, ...
                        trainLabel, ...
                        'Surrogate', 'on');                    
                        %  'SplitCriterion', 'gdi', ...
                        %  'MaxNumSplits', 20, ...
                        %  'MaxNumSplits', 50, ...
                    prediction_BDT = predict(model, testData);
                    trPred_BDT = predict(model, trainData);                
                    
                    %t = confusionmatStats(trainLabel, trPred_BDT);
                    %fprintf('BDT-trainAccuracy - %f\n', t.accuracy);             
                    %t = confusionmatStats(testLabel, prediction_BDT);
                    %fprintf('BDT-testAccuracy - %f\n', t.accuracy);
                end        
            end

            %Accuracy of each fold
            options = unique(testLabel);
            testIndex2 = [testIndex2; testIndex1];
            tr_votes = [];
            votes = [];
            w = [];
            
            for j = 1:1:size(classification_methods,2)
                if(strcmpi(classification_methods{j},'Adaboost'))
                    t = confusionmatStats(testLabel, prediction_Adaboost);
                    t.no = k;
                    t.score = [prediction_Adaboost testLabel testIndex1];
                    tempStats_Adaboost = [tempStats_Adaboost; t];

                    newlabel_Adaboost(testIndex,:) = prediction_Adaboost; 
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_Adaboost];             
                        tr_votes = [tr_votes trPred_Adaboost];
                        w =[w t.accuracy];
                    end
                end
                if(strcmpi(classification_methods{j},'KNN'))
                    t = confusionmatStats(testLabel, prediction_KNN);
                    t.no = k;
                    t.score = [prediction_KNN testLabel testIndex1];
                    tempStats_KNN = [tempStats_KNN; t];

                    newlabel_KNN(testIndex,:) = prediction_KNN;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_KNN];            
                        tr_votes = [tr_votes trPred_KNN];
                        w =[w t.accuracy];
                    end
                end            
                if(strcmpi(classification_methods{j},'SVM'))
                    t = confusionmatStats(testLabel, prediction_SVM);
                    t.no = k;
                    t.score = [prediction_SVM testLabel testIndex1];
                    tempStats_SVM = [tempStats_SVM; t];

                    newlabel_SVM(testIndex,:) = prediction_SVM;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_SVM];
                        tr_votes = [tr_votes trPred_SVM];
                        w =[w t.accuracy];
                    end
                end                    
                if(strcmpi(classification_methods{j},'RandomForest'))
                    t = confusionmatStats(testLabel, prediction_RandomForest);
                    t.no = k;
                    t.score = [prediction_RandomForest testLabel testIndex1];
                    tempStats_RandomForest = [tempStats_RandomForest; t];

                    newlabel_RandomForest(testIndex,:) = prediction_RandomForest;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_RandomForest];            
                        tr_votes = [tr_votes trPred_RandomForest];
                        w =[w t.accuracy];
                    end
                end                            
                if(strcmpi(classification_methods{j},'Bagged'))
                    t = confusionmatStats(testLabel, prediction_Bagged);
                    t.no = k;
                    t.score = [prediction_Bagged testLabel testIndex1];
                    tempStats_Bagged = [tempStats_Bagged; t];

                    newlabel_Bagged(testIndex,:) = prediction_Bagged;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_Bagged];            
                        tr_votes = [tr_votes trPred_Bagged];
                        w =[w t.accuracy];
                    end
                end            
                if(strcmpi(classification_methods{j},'DNN'))
                    t = confusionmatStats(testLabelDNN, prediction_DNN);
                    t.no = k;
                    t.score = [prediction_DNN testLabelDNN testIndex1];
                    tempStats_DNN = [tempStats_DNN; t];

                    newlabel_DNN(testIndex,:) = prediction_DNN;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_DNN];            
                        tr_votes = [tr_votes trPred_DNN];
                        w =[w t.accuracy];
                    end
                end             
                if(strcmpi(classification_methods{j},'NN'))
                    t = confusionmatStats(testLabel, prediction_NN);
                    t.no = k;
                    t.score = [prediction_NN testLabel testIndex1];
                    tempStats_NN = [tempStats_NN; t];

                    newlabel_NN(testIndex,:) = prediction_NN;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_NN];            
                        tr_votes = [tr_votes trPred_NN];
                        w =[w t.accuracy];
                    end
                end            
                if(strcmpi(classification_methods{j},'BDT'))
                    t = confusionmatStats(testLabel, prediction_BDT);
                    t.no = k;
                    t.score = [prediction_BDT testLabel testIndex1];
                    tempStats_BDT = [tempStats_BDT; t];

                    newlabel_BDT(testIndex,:) = prediction_BDT;
                    if(strcmpi(isinFusion{j}, '1'))
                        votes = [votes prediction_BDT];            
                        tr_votes = [tr_votes trPred_BDT];
                        w =[w t.accuracy];
                    end
                end
            end

            %Fusion
            if(fusion == 1)     
                prediction_WMV = weighted_majority_voting(options, votes, w);
                newlabel_WMV(testIndex,:) = prediction_WMV;
                prediction_NB = nb_combiner(votes,tr_votes,trainLabel);
                newlabel_NB(testIndex,:) = prediction_NB;
                prediction_BKS = bks_combiner(votes,tr_votes,trainLabel,options,w);
                newlabel_BKS(testIndex,:) = prediction_BKS;

                t = confusionmatStats(testLabel, prediction_WMV);
                t.no = k;
                t.score = [prediction_WMV testLabel testIndex1];
                t.classification_methods=classification_methods;
                tempStats_WMV = [tempStats_WMV; t];


                t = confusionmatStats(testLabel, prediction_NB);
                t.no = k;
                t.score = [prediction_NB testLabel testIndex1];
                t.classification_methods=classification_methods;
                tempStats_NB = [tempStats_NB; t];


                t = confusionmatStats(testLabel, prediction_BKS);
                t.no = k;
                t.score = [prediction_BKS testLabel testIndex1];
                t.classification_methods=classification_methods;
                tempStats_BKS = [tempStats_BKS; t];

            end
        end

        %Combining Results
        class_names = {}'; 
        final_acc = [];

        for j = 1:1:size(classification_methods,2)
            if(strcmpi(classification_methods{j},'Adaboost'))            
                stat.foldstats.Adaboost=tempStats_Adaboost;

                if(loop == 1)
                    stat.finalstats.Adaboost = stat.foldstats.Adaboost;
                else
                    t = confusionmatStats(label, newlabel_Adaboost);
                    t.score = [newlabel_Adaboost label testIndex2];
                    stat.finalstats.Adaboost = t;
                end            
                class_names = [class_names 'Adaboost'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end
            if(strcmpi(classification_methods{j},'KNN'))          
                stat.foldstats.KNN=tempStats_KNN;

                if(loop == 1)
                    stat.finalstats.KNN = stat.foldstats.KNN;
                else
                    t = confusionmatStats(label, newlabel_KNN);
                    t.score = [newlabel_KNN label testIndex2];
                    stat.finalstats.KNN = t;
                end
                class_names = [class_names 'KNN'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end            
            if(strcmpi(classification_methods{j},'SVM'))          
                stat.foldstats.SVM=tempStats_SVM;

                if(loop == 1)
                    stat.finalstats.SVM = stat.foldstats.SVM;
                else
                    t = confusionmatStats(label, newlabel_SVM);
                    t.score = [newlabel_SVM label testIndex2];
                    stat.finalstats.SVM = t;
                end
                class_names = [class_names 'SVM'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end                 
            if(strcmpi(classification_methods{j},'RandomForest'))          
                stat.foldstats.RandomForest=tempStats_RandomForest;

                if(loop == 1)
                    stat.finalstats.RandomForest = stat.foldstats.RandomForest;
                else
                    t = confusionmatStats(label, newlabel_RandomForest);
                    t.score = [newlabel_RandomForest label testIndex2];
                    stat.finalstats.RandomForest = t;
                end
                class_names = [class_names 'RandomForest'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end                  
            if(strcmpi(classification_methods{j},'Bagged'))          
                stat.foldstats.Bagged=tempStats_Bagged;

                if(loop == 1)
                    stat.finalstats.Bagged = stat.foldstats.Bagged;
                else
                    t = confusionmatStats(label, newlabel_Bagged);
                    t.score = [newlabel_Bagged label testIndex2];
                    stat.finalstats.Bagged = t;
                end
                class_names = [class_names 'Bagged'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end                 
            if(strcmpi(classification_methods{j},'DNN'))          
                stat.foldstats.DNN=tempStats_DNN;

                if(loop == 1)
                    stat.finalstats.DNN = stat.foldstats.DNN;
                else
                    t = confusionmatStats(label, newlabel_DNN);
                    t.score = [newlabel_DNN label testIndex2];
                    stat.finalstats.DNN = t;
                end
                class_names = [class_names 'DNN'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end                   
            if(strcmpi(classification_methods{j},'NN'))          
                stat.foldstats.NN=tempStats_NN;

                if(loop == 1)
                    stat.finalstats.NN = stat.foldstats.NN;
                else
                    t = confusionmatStats(label, newlabel_NN);
                    t.score = [newlabel_NN label testIndex2];
                    stat.finalstats.NN = t;
                end
                class_names = [class_names 'NN'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end            
            if(strcmpi(classification_methods{j},'BDT'))          
                stat.foldstats.BDT=tempStats_BDT;

                if(loop == 1)
                    stat.finalstats.BDT = stat.foldstats.BDT;
                else
                    t = confusionmatStats(label, newlabel_BDT);
                    t.score = [newlabel_BDT label testIndex2];
                    stat.finalstats.BDT = t;
                end
                class_names = [class_names 'BDT'];
                final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
            end
        end
        if(fusion == 1)  
            stat.foldstats.WMV_Fusion=tempStats_WMV;            
            if(loop == 1)
                stat.finalstats.WMV_Fusion = stat.foldstats.WMV_Fusion;
            else
                t = confusionmatStats(label, newlabel_WMV);
                t.score = [newlabel_WMV label testIndex2];
                stat.finalstats.WMV_Fusion = t;
            end
            class_names = [class_names 'WMV_Fusion'];
            final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];

            stat.foldstats.NB_Fusion=tempStats_NB;            
            if(loop == 1)
                stat.finalstats.NB_Fusion = stat.foldstats.NB_Fusion;
            else
                t = confusionmatStats(label, newlabel_NB);
                t.score = [newlabel_NB label testIndex2];
                stat.finalstats.NB_Fusion = t;
            end
            class_names = [class_names 'NB_Fusion'];
            final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];

            stat.foldstats.BKS_Fusion=tempStats_BKS;            
            if(loop == 1)
                stat.finalstats.BKS_Fusion = stat.foldstats.BKS_Fusion;
            else
                t = confusionmatStats(label, newlabel_BKS);
                t.score = [newlabel_BKS label testIndex2];
                stat.finalstats.BKS_Fusion = t;
            end
            class_names = [class_names 'BKS_Fusion'];
            final_acc = [final_acc t.Fscore(length(t.Fscore)-1)];
        end

        %Analysis
        [pathstr,name,ext] = fileparts(fileName);     
        save([strcat(pathName,name,'_Stat') '.mat'], 'stat');
        ok = performance_analysis(stat, class_names', final_acc', pathName, strcat(name,'_RES'));

        o=o+1;
        
       beep;
    end
end
