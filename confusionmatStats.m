function stats = confusionmatStats(group,grouphat)
% INPUT
% group = true class labels
% grouphat = predicted class labels
%
% OR INPUT
% stats = confusionmatStats(group);
% group = confusion matrix from matlab function (confusionmat)
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____| 
%       Actual  p |     |     |
%      Classes  n |     |     |
%
% stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
% stats.precision = TP / (TP + FP)                  % for each class label
% stats.sensitivity = TP / (TP + FN)                % for each class label
% stats.specificity = TN / (FP + TN)                % for each class label
% stats.recall = sensitivity                        % for each class label
% stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 

order=0;
field1 = 'confusionMat';
if nargin < 2
    value1 = group;
else
    [value1,order] = confusionmat(group,grouphat);
end

numOfClasses = size(order,1);
if(numOfClasses ~= 8)
    check=1;
    beep;
end
totalSamples = sum(sum(value1));
    
%field2 = 'accuracy';  value2 = (2*trace(value1)+sum(sum(2*value1)))/(numOfClasses*totalSamples);

[TP,TN,FP,FN] = deal(zeros(numOfClasses,1));
[sensitivity,specificity,precision,f_score,class_accuracy] = deal(zeros(numOfClasses+2,1));
for class = 1:numOfClasses
   TP(class) = value1(class,class);
   tempMat = value1;
   tempMat(:,class) = []; % remove column
   tempMat(class,:) = []; % remove row
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(value1(:,class))-TP(class);
   FN(class) = sum(value1(class,:))-TP(class);
end
field2 = 'accuracy';  value2 = (sum(TP)+sum(TN))/(sum(TP)+sum(TN)+sum(FP)+sum(FN));
field13 = 'accuracy2';  value13 = (sum(TP))/size(group,1);


for class = 1:numOfClasses
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    specificity(class) = TN(class) / (FP(class) + TN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
    class_accuracy(class) = (TP(class) + TN(class))/(TP(class) + TN(class) + FP(class) + FN(class));
end
sensitivity(numOfClasses+1)=mean(sensitivity(1:numOfClasses));
specificity(numOfClasses+1)=mean(specificity(1:numOfClasses));
precision(numOfClasses+1)=mean(precision(1:numOfClasses));
f_score(numOfClasses+1)=mean(f_score(1:numOfClasses));

class_accuracy(numOfClasses+1)=mean(class_accuracy(1:numOfClasses));

sensitivity(numOfClasses+2)=std(sensitivity(1:numOfClasses));
specificity(numOfClasses+2)=std(specificity(1:numOfClasses));
precision(numOfClasses+2)=std(precision(1:numOfClasses));
f_score(numOfClasses+2)=std(f_score(1:numOfClasses));
class_accuracy(numOfClasses+2)=std(class_accuracy(1:numOfClasses));

field14 = 'class_accuracy';  value14 = class_accuracy;
field3 = 'sensitivity';  value3 = sensitivity;
field4 = 'specificity';  value4 = specificity;
field5 = 'precision';  value5 = precision;
field6 = 'recall';  value6 = sensitivity;
field7 = 'Fscore';  value7 = f_score;
field8 = 'TruePos';  value8 = TP;
field9 = 'TrueNeg';  value9 = TN;
field10 = 'FalsePos';  value10 = FP;
field11 = 'FalseNeg';  value11 = FN;
field12 = 'Order';  value12 = order;

valuenew=[order value1];
order1=[-999 order'];
valuenew=[order1 ; valuenew];

stats = struct(field1,valuenew,field2,value2,field13,value13,field14,value14,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7,field8,value8,field9,value9,field10,value10,field11,value11,field12,value12);

% a=[stat.finalstats.KNN.precision(size(stat.finalstats.KNN.precision,1)-1) stat.finalstats.KNN.recall(size(stat.finalstats.KNN.recall,1)-1) stat.finalstats.KNN.Fscore(size(stat.finalstats.KNN.Fscore,1)-1);
%    stat.finalstats.BDT.precision(size(stat.finalstats.BDT.precision,1)-1) stat.finalstats.BDT.recall(size(stat.finalstats.BDT.recall,1)-1) stat.finalstats.BDT.Fscore(size(stat.finalstats.BDT.Fscore,1)-1);
%    stat.finalstats.DNN.precision(size(stat.finalstats.DNN.precision,1)-1) stat.finalstats.DNN.recall(size(stat.finalstats.DNN.recall,1)-1) stat.finalstats.DNN.Fscore(size(stat.finalstats.DNN.Fscore,1)-1);
%    stat.finalstats.Adaboost.precision(size(stat.finalstats.Adaboost.precision,1)-1) stat.finalstats.Adaboost.recall(size(stat.finalstats.Adaboost.recall,1)-1) stat.finalstats.Adaboost.Fscore(size(stat.finalstats.Adaboost.Fscore,1)-1);
%    stat.finalstats.WMV_Fusion.precision(size(stat.finalstats.WMV_Fusion.precision,1)-1) stat.finalstats.WMV_Fusion.recall(size(stat.finalstats.WMV_Fusion.recall,1)-1) stat.finalstats.WMV_Fusion.Fscore(size(stat.finalstats.WMV_Fusion.Fscore,1)-1);
%    stat.finalstats.NB_Fusion.precision(size(stat.finalstats.NB_Fusion.precision,1)-1) stat.finalstats.NB_Fusion.recall(size(stat.finalstats.NB_Fusion.recall,1)-1) stat.finalstats.NB_Fusion.Fscore(size(stat.finalstats.NB_Fusion.Fscore,1)-1);
%    stat.finalstats.BKS_Fusion.precision(size(stat.finalstats.BKS_Fusion.precision,1)-1) stat.finalstats.BKS_Fusion.recall(size(stat.finalstats.BKS_Fusion.recall,1)-1) stat.finalstats.BKS_Fusion.Fscore(size(stat.finalstats.BKS_Fusion.Fscore,1)-1);
% ];
% a=a.*100;


% 
% y = stat.finalstats.BKS_Fusion.score(:,1);
% x = stat.finalstats.BKS_Fusion.score(:,2);
% 
% temp = x';
% [c, ia, ic] = unique(temp,'sorted');
% newx = zeros(size(c,2), size(x,1));
% for i=1:1:size(c,2)
%     index = find(temp==c(1,i));
%     newx(i,index)=1;
% end
% 
% temp = y';
% [c, ia, ic] = unique(temp,'sorted');
% newy = zeros(size(c,2), size(y,1));
% for i=1:1:size(c,2)
%     index = find(temp==c(1,i));
%     newy(i,index)=1;
% end
% 
% plotconfusion(newx,newy);