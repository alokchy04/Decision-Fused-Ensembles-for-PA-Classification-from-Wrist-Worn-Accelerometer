function [data, dataHeader, error] = read_CSV_File(fullpath)
%read_CSV_File Summary of this function goes here
%   this funtion take the fullpath of the File and do necessary conversion
%   and return data in desired format data
% FOR MAPPING READ
    
    dataHeader = 0;
    data = 0;
    error = 0;
    try        
        quote = '"';
        sep = ',';
        escape = '\';
        [numbers, text] = swallow_csv(fullpath, quote, sep, escape);

        %first line will be header
        dataHeader = text(1,1:size(text,2));
        data = numbers(2:size(numbers,1),1:size(numbers,2));

    catch Ex1        
        error = 2;
    end
      
    if (error == 2)
        f = warndlg('ERROR - Coudn''t load! Problem Opening File', 'Failed');
        drawnow();
        waitfor(f);
    end
end

function names = parse_line(tempstr)
    ch = char(44); 
    index = 0;
    while length(tempstr > 0) % separate objects based on delimiter positions
         index = index + 1;
         tempstr = fliplr(deblank(fliplr(deblank(tempstr))));
         if isempty(find(tempstr == ch))
              names{index} = tempstr;
              tempstr = [];
         else
              names{index} = tempstr(1:find(tempstr == ch)-1);
              tempstr(1:find(tempstr == ch)) = [];
         end  
    end
    return
end