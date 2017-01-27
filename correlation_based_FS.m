function [newfet, newfetHeader, newlabel, minCorr, numoffeatures] = correlation_based_FS(fetData, fetHeader, label, howMany, threshold)   
    
    if((howMany <=0 || howMany > size(fetData,2))) 
        howMany = size(fetData,2);
    end
    
    total = [fetData label];
    
    %remove NaN Features    
    nanIndicator = ~any(isnan(total),2);
    total = total(nanIndicator,:);
    fetData = total(:,1:size(total,2)-size(label,2));
    newlabel = total(:,size(total,2)-size(label,2)+1:size(total,2));
    
    mycorr = corrcoef(total);    
    impRow = abs(mycorr(size(mycorr,1),1:size(mycorr,2)-size(label,2)));
    
    newfet = [];
    newfetHeader = {};    
    temp= sort(impRow, 'descend');  
    
    numoffeatures = size(impRow,2);
    for i=1:1:size(impRow,2)
        if(threshold ~= 0 && temp(1,i) < threshold)
            numoffeatures = i-1;
            break;
        end
        if(howMany ~= 0 && i > howMany)
            numoffeatures = i-1;
            break;
        end
        
        minCorr = temp(1,i);
        k = find(impRow == temp(1,i));

        newfet = [newfet fetData(:,k(1))];
        newfetHeader{i} = fetHeader{k(1)}; 
        impRow(1,k(1)) = -999;
    end   
end