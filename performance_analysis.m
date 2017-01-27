function ok = performance_analysis(stat, class_names, final_acc, pathname, filename)

    myvar=stat;
    
    [~, idx]=sort(final_acc, 'descend');
    s_final = [class_names(idx,:) num2cell(final_acc(idx,:))];

    ok = -1; 
    fprintf('Final Outcome:\n');
    for i=1:1:size(s_final,1)
        if(i~=size(s_final,1))
            fprintf('%d) %s %5.4f%% > ',i, s_final{i,1}, s_final{i,2}*100);
        else
            fprintf('%d) %s %5.4f%%\n',i, s_final{i,1}, s_final{i,2}*100);
        end
        
        if(strcmpi(s_final{i,1},'RandomForest') || strcmpi(s_final{i,1},'Bagged') || strcmpi(s_final{i,1},'Adaboost'))
        else
            if(ok == -1)
                if(strcmpi(s_final{i,1},'WMV_Fusion') || strcmpi(s_final{i,1},'NB_Fusion') || strcmpi(s_final{i,1},'BKS_Fusion'))
                    ok = 1;
                else
                    ok = 0;
                end
            end
        end
        
    end
    
    nooffold = myvar.nooffold;
    if(myvar.nooffold == 0)
        nooffold = 1;
    end
    
    header={};
    tbl_acc=[];
    tbl_pre=[];
    tbl_rcl=[];
    tbl_fs=[];
    for j=1:1:size(class_names,1)
        if(strcmpi(class_names{j}, 'KNN'))
            acc_KNN=[myvar.finalstats.KNN.class_accuracy];
            acc_KNN = acc_KNN.*100;
            
            pre_KNN=[myvar.finalstats.KNN.precision];
            pre_KNN = pre_KNN.*100;
            
            rcl_KNN=[myvar.finalstats.KNN.recall];
            rcl_KNN = rcl_KNN.*100;
                        
            fs_KNN=[myvar.finalstats.KNN.Fscore];
            fs_KNN = fs_KNN.*100; 
                       
            header = [header 'KNN'];
            tbl_acc = [tbl_acc acc_KNN];
            tbl_pre = [tbl_pre pre_KNN];
            tbl_rcl = [tbl_rcl rcl_KNN];
            tbl_fs = [tbl_fs fs_KNN];
        end        

        if(strcmpi(class_names{j}, 'NN'))
            acc_NN=[myvar.finalstats.NN.class_accuracy];
            acc_NN = acc_NN.*100;
            
            pre_NN=[myvar.finalstats.NN.precision];
            pre_NN = pre_NN.*100;
            
            rcl_NN=[myvar.finalstats.NN.recall];
            rcl_NN = rcl_NN.*100;
                        
            fs_NN=[myvar.finalstats.NN.Fscore];
            fs_NN = fs_NN.*100; 
                       
            header = [header 'NN'];
            tbl_acc = [tbl_acc acc_NN];
            tbl_pre = [tbl_pre pre_NN];
            tbl_rcl = [tbl_rcl rcl_NN];
            tbl_fs = [tbl_fs fs_NN];
        end

        if(strcmpi(class_names{j}, 'SVM'))
            acc_SVM=[myvar.finalstats.SVM.class_accuracy];
            acc_SVM = acc_SVM.*100;
            
            pre_SVM=[myvar.finalstats.SVM.precision];
            pre_SVM = pre_SVM.*100;
            
            rcl_SVM=[myvar.finalstats.SVM.recall];
            rcl_SVM = rcl_SVM.*100;
                        
            fs_SVM=[myvar.finalstats.SVM.Fscore];
            fs_SVM = fs_SVM.*100; 
                       
            header = [header 'SVM'];
            tbl_acc = [tbl_acc acc_SVM];
            tbl_pre = [tbl_pre pre_SVM];
            tbl_rcl = [tbl_rcl rcl_SVM];
            tbl_fs = [tbl_fs fs_SVM];
        end
        
        if(strcmpi(class_names{j}, 'RandomForest'))
            
            acc_RandomForest=[myvar.finalstats.RandomForest.class_accuracy];
            acc_RandomForest = acc_RandomForest.*100;
            
            pre_RandomForest=[myvar.finalstats.RandomForest.precision];
            pre_RandomForest = pre_RandomForest.*100;
            
            rcl_RandomForest=[myvar.finalstats.RandomForest.recall];
            rcl_RandomForest = rcl_RandomForest.*100;
                        
            fs_RandomForest=[myvar.finalstats.RandomForest.Fscore];
            fs_RandomForest = fs_RandomForest.*100; 
                       
            header = [header 'RandomForest'];
            tbl_acc = [tbl_acc acc_RandomForest];
            tbl_pre = [tbl_pre pre_RandomForest];
            tbl_rcl = [tbl_rcl rcl_RandomForest];
            tbl_fs = [tbl_fs fs_RandomForest];
        end
        
        if(strcmpi(class_names{j}, 'Bagged'))
            acc_Bagged=[myvar.finalstats.Bagged.class_accuracy];
            acc_Bagged = acc_Bagged.*100;
            
            pre_Bagged=[myvar.finalstats.Bagged.precision];
            pre_Bagged = pre_Bagged.*100;
            
            rcl_Bagged=[myvar.finalstats.Bagged.recall];
            rcl_Bagged = rcl_Bagged.*100;
                        
            fs_Bagged=[myvar.finalstats.Bagged.Fscore];
            fs_Bagged = fs_Bagged.*100; 
                       
            header = [header 'Bagged'];
            tbl_acc = [tbl_acc acc_Bagged];
            tbl_pre = [tbl_pre pre_Bagged];
            tbl_rcl = [tbl_rcl rcl_Bagged];
            tbl_fs = [tbl_fs fs_Bagged];
        end
        
        if(strcmpi(class_names{j}, 'BDT'))
            acc_BDT=[myvar.finalstats.BDT.class_accuracy];
            acc_BDT = acc_BDT.*100;
            
            pre_BDT=[myvar.finalstats.BDT.precision];
            pre_BDT = pre_BDT.*100;
            
            rcl_BDT=[myvar.finalstats.BDT.recall];
            rcl_BDT = rcl_BDT.*100;
                        
            fs_BDT=[myvar.finalstats.BDT.Fscore];
            fs_BDT = fs_BDT.*100; 
                       
            header = [header 'BDT'];
            tbl_acc = [tbl_acc acc_BDT];
            tbl_pre = [tbl_pre pre_BDT];
            tbl_rcl = [tbl_rcl rcl_BDT];
            tbl_fs = [tbl_fs fs_BDT];
        end    

        if(strcmpi(class_names{j}, 'DNN'))
            acc_DNN=[myvar.finalstats.DNN.class_accuracy];
            acc_DNN = acc_DNN.*100;
            
            pre_DNN=[myvar.finalstats.DNN.precision];
            pre_DNN = pre_DNN.*100;
            
            rcl_DNN=[myvar.finalstats.DNN.recall];
            rcl_DNN = rcl_DNN.*100;
                        
            fs_DNN=[myvar.finalstats.DNN.Fscore];
            fs_DNN = fs_DNN.*100; 
                       
            header = [header 'DNN'];
            tbl_acc = [tbl_acc acc_DNN];
            tbl_pre = [tbl_pre pre_DNN];
            tbl_rcl = [tbl_rcl rcl_DNN];
            tbl_fs = [tbl_fs fs_DNN];
        end

        if(strcmpi(class_names{j}, 'Adaboost'))
            acc_Adaboost=[myvar.finalstats.Adaboost.class_accuracy];
            acc_Adaboost = acc_Adaboost.*100;
            
            pre_Adaboost=[myvar.finalstats.Adaboost.precision];
            pre_Adaboost = pre_Adaboost.*100;
            
            rcl_Adaboost=[myvar.finalstats.Adaboost.recall];
            rcl_Adaboost = rcl_Adaboost.*100;
                        
            fs_Adaboost=[myvar.finalstats.Adaboost.Fscore];
            fs_Adaboost = fs_Adaboost.*100; 
                       
            header = [header 'Adaboost'];
            tbl_acc = [tbl_acc acc_Adaboost];
            tbl_pre = [tbl_pre pre_Adaboost];
            tbl_rcl = [tbl_rcl rcl_Adaboost];
            tbl_fs = [tbl_fs fs_Adaboost];
        end

        if(strcmpi(class_names{j}, 'WMV_Fusion'))
            acc_WMV_Fusion=[myvar.finalstats.WMV_Fusion.class_accuracy];
            acc_WMV_Fusion = acc_WMV_Fusion.*100;
            
            pre_WMV_Fusion=[myvar.finalstats.WMV_Fusion.precision];
            pre_WMV_Fusion = pre_WMV_Fusion.*100;
            
            rcl_WMV_Fusion=[myvar.finalstats.WMV_Fusion.recall];
            rcl_WMV_Fusion = rcl_WMV_Fusion.*100;
                        
            fs_WMV_Fusion=[myvar.finalstats.WMV_Fusion.Fscore];
            fs_WMV_Fusion = fs_WMV_Fusion.*100; 
                       
            header = [header 'WMV_Fusion'];
            tbl_acc = [tbl_acc acc_WMV_Fusion];
            tbl_pre = [tbl_pre pre_WMV_Fusion];
            tbl_rcl = [tbl_rcl rcl_WMV_Fusion];
            tbl_fs = [tbl_fs fs_WMV_Fusion];
        end

        if(strcmpi(class_names{j}, 'NB_Fusion'))
            acc_NB_Fusion=[myvar.finalstats.NB_Fusion.class_accuracy];
            acc_NB_Fusion = acc_NB_Fusion.*100;
            
            pre_NB_Fusion=[myvar.finalstats.NB_Fusion.precision];
            pre_NB_Fusion = pre_NB_Fusion.*100;
            
            rcl_NB_Fusion=[myvar.finalstats.NB_Fusion.recall];
            rcl_NB_Fusion = rcl_NB_Fusion.*100;
                        
            fs_NB_Fusion=[myvar.finalstats.NB_Fusion.Fscore];
            fs_NB_Fusion = fs_NB_Fusion.*100; 
                       
            header = [header 'NB_Fusion'];
            tbl_acc = [tbl_acc acc_NB_Fusion];
            tbl_pre = [tbl_pre pre_NB_Fusion];
            tbl_rcl = [tbl_rcl rcl_NB_Fusion];
            tbl_fs = [tbl_fs fs_NB_Fusion];     
        end

        if(strcmpi(class_names{j}, 'BKS_Fusion'))
            acc_BKS_Fusion=[myvar.finalstats.BKS_Fusion.class_accuracy];
            acc_BKS_Fusion = acc_BKS_Fusion.*100;
            
            pre_BKS_Fusion=[myvar.finalstats.BKS_Fusion.precision];
            pre_BKS_Fusion = pre_BKS_Fusion.*100;
            
            rcl_BKS_Fusion=[myvar.finalstats.BKS_Fusion.recall];
            rcl_BKS_Fusion = rcl_BKS_Fusion.*100;
                        
            fs_BKS_Fusion=[myvar.finalstats.BKS_Fusion.Fscore];
            fs_BKS_Fusion = fs_BKS_Fusion.*100; 
                       
            header = [header 'BKS_Fusion'];
            tbl_acc = [tbl_acc acc_BKS_Fusion];
            tbl_pre = [tbl_pre pre_BKS_Fusion];
            tbl_rcl = [tbl_rcl rcl_BKS_Fusion];
            tbl_fs = [tbl_fs fs_BKS_Fusion];  
        end
    end
    
    header = ['Activity' header];
    act_class=[];
    for i=1:1:size(tbl_acc,1)
        act_class=[act_class;i];
    end
    act_class=num2cell(act_class);
    act_class{size(tbl_acc,1)-1}='mean';
    act_class{size(tbl_acc,1)}='std';
    
    tbl_acc = [act_class num2cell(tbl_acc)];
    tbl_pre = [act_class num2cell(tbl_pre)];
    tbl_rcl = [act_class num2cell(tbl_rcl)];
    tbl_fs = [act_class num2cell(tbl_fs)]; 
    
    
    printresults(strcat(pathname,['RES_acuracy_' filename '.csv']), tbl_acc, header);
    printresults(strcat(pathname,['RES_precision_' filename '.csv']), tbl_pre, header);
    printresults(strcat(pathname,['RES_recall_' filename '.csv']), tbl_rcl, header);
    printresults(strcat(pathname,['RES_fscore_' filename '.csv']), tbl_fs, header);
end



function printresults(filename,m,headers,r,c)

    %% initial checks on the inputs
    if ~ischar(filename)
        error('FILENAME must be a string');
    end

    % the r and c inputs are optional and need to be filled in if they are
    % missing
    if nargin < 4
        r = 0;
    end
    if nargin < 5
        c = 0;
    end

    if ~iscellstr(headers)
        error('Header must be cell array of strings')
    end


    if length(headers) ~= size(m,2)
        error('number of header entries must match the number of columns in the data')
    end

    %% write the header string to the file

    %turn the headers into a single comma seperated string if it is a cell
    %array, 
    header_string = headers{1};
    for i = 2:length(headers)
        header_string = [header_string,',',headers{i}];
    end
    %if the data has an offset shifting it right then blank commas must
    %be inserted to match
    if r>0
        for i=1:r
            header_string = [',',header_string];
        end
    end

    %write the string to a file
    fid = fopen(filename,'w+');
        fprintf(fid,'%s\r\n',header_string);

        if(iscell(m)==1)
            [rows, cols]=size(m);
            for i=1:rows-2
                  fprintf(fid,'%5.2f,',m{i,1:end-1});
                  fprintf(fid,'%5.2f\r\n',m{i,end});
            end
            for i=rows-1:rows
                  fprintf(fid,'%s,',m{i,1});
                  fprintf(fid,'%5.2f,',m{i,2:end-1});
                  fprintf(fid,'%5.2f\r\n',m{i,end});
            end
        end
    fclose(fid);

    %% write the append the data to the file

    %
    % Call dlmwrite with a comma as the delimiter
    %
    if(iscell(m)==0)
        dlmwrite(filename, m,'-append', 'precision','%.15f', 'delimiter',',','roffset', r,'coffset',c);
    end
end