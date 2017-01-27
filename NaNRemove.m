folderName = uigetdir;
try
    if(folderName == 0)
        return;
    end
catch
end

sourceFiles = dir(fullfile(folderName, '*.csv'));
for i=1:1:size(sourceFiles,1)
    fileName{i}=sourceFiles(i).name;
    pathName{i}=strcat(folderName,'\');
    
    quote = '"';
    sep = ',';
    escape = '\';
    [numbers, text] = swallow_csv(strcat(pathName{i}, fileName{i}), quote, sep, escape);
    
    data=numbers(2:end,:);    
    data=data(~any(isnan(data),2),:);    
    dataHeader = text(1,2:size(text,2));
    
    mkdir(pathName{i},'newHR');
    csvwrite_with_headers(strcat(pathName{i},'newHR\',fileName{i}), data, ['Datetime' dataHeader]);
    
    fprintf('Done %d', i);
end
fprintf('Finally Done');
