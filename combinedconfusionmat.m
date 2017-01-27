% a=stat.finalstats.RandomForest.confusionMat(2:end,2:end);
% b=stat.finalstats.Bagged.confusionMat(2:end,2:end);
% c=stat.finalstats.Adaboost.confusionMat(2:end,2:end);
% d=stat.finalstats.WMV_Fusion.confusionMat(2:end,2:end);
% e=stat.finalstats.NB_Fusion.confusionMat(2:end,2:end);
% f=stat.finalstats.BKS_Fusion.confusionMat(2:end,2:end);
% 
% for i=1:1:size(a,1)
%     if i== 1
%         final = [a(1,:);b(1,:);c(1,:);d(1,:);e(1,:);f(1,:)];
%     else
%         final = [final; a(i,:);b(i,:);c(i,:);d(i,:);e(i,:);f(i,:)];
%     end
% end
% 
% ans=final;


%======================================


% folderName = '\\qut.edu.au\Documents\StudentHome\Group32$\n8485232\Desktop\1 FS & Classification\FET - LinearSVM\fet'; %uigetdir;
% try
%     if(folderName == 0)
%         return;
%     end
% catch
% end  
% sourceFiles = dir(fullfile(folderName, '*.csv'));
% 
% for k=1:1:3 %dataset
%     j = 0;
%     for i=1:1:size(sourceFiles,1)
%         if(str2num(sourceFiles(i).name(length(sourceFiles(i).name)-4))==k)
%             j = j + 1;
%             [~, fetHeader, ~] = read_CSV_File(strcat(folderName,'\',sourceFiles(i).name));
%             fetHeader = fetHeader(1:end-1);
%             
%             if j==1
%                 itsct{1,k} = fetHeader;
%             else
%                 itsct{1,k} = intersect(itsct{1,k},fetHeader);
%             end
%         end
%     end
%     itsct{1,k} = sort(itsct{1,k});
% end
% res=itsct;



for i=1:1:size(stat.foldstats.RandomForest)
    if i==1
        RandomForest=stat.foldstats.RandomForest(i).Fscore(end-1);
        Bagged=stat.foldstats.Bagged(i).Fscore(end-1);
        Adaboost=stat.foldstats.Adaboost(i).Fscore(end-1);
        BDT=stat.foldstats.BDT(i).Fscore(end-1);
        KNN=stat.foldstats.KNN(i).Fscore(end-1);
        SVM=stat.foldstats.SVM(i).Fscore(end-1);
        NN=stat.foldstats.NN(i).Fscore(end-1);
        WMV_Fusion=stat.foldstats.WMV_Fusion(i).Fscore(end-1);
        NB_Fusion=stat.foldstats.NB_Fusion(i).Fscore(end-1);
        BKS_Fusion=stat.foldstats.BKS_Fusion(i).Fscore(end-1);
    else
        RandomForest=[RandomForest; stat.foldstats.RandomForest(i).Fscore(end-1)];
        Bagged=[Bagged; stat.foldstats.Bagged(i).Fscore(end-1)];
        Adaboost=[Adaboost; stat.foldstats.Adaboost(i).Fscore(end-1)];
        BDT=[BDT; stat.foldstats.BDT(i).Fscore(end-1)];
        KNN=[KNN; stat.foldstats.KNN(i).Fscore(end-1)];
        SVM=[SVM; stat.foldstats.SVM(i).Fscore(end-1)];
        NN=[NN; stat.foldstats.NN(i).Fscore(end-1)];
        WMV_Fusion=[WMV_Fusion; stat.foldstats.WMV_Fusion(i).Fscore(end-1)];
        NB_Fusion=[NB_Fusion; stat.foldstats.NB_Fusion(i).Fscore(end-1)];
        BKS_Fusion=[BKS_Fusion; stat.foldstats.BKS_Fusion(i).Fscore(end-1)];
    end
end
res= [RandomForest Bagged Adaboost BDT KNN SVM NN WMV_Fusion NB_Fusion BKS_Fusion];
res = res;