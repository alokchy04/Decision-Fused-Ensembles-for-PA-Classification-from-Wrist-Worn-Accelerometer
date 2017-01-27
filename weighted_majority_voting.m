function newlabel = weighted_majority_voting(options, votes, w)
%     w=[7 2 6];
%     votes = [1 2 2
%          3 1 1
%          4 2 1
%          1 1 3];
%     options = [1,2,3,4]';    

%'//Make a cube of the options that is number of options by m by n
    OPTIONS = repmat(options, [1, size(w, 2), size(votes, 1)]);

    %//Compare the votes (streched to make surface) against a uniforma surface of each option
    B = bsxfun(@eq, permute(votes, [3 2 1]) ,OPTIONS);

    %//Find a weighted sum
    W = squeeze(sum(bsxfun(@times, repmat(w, size(options, 1), 1), B), 2))';

    %'//Find the options with the highest weighted sum
    [xx, i] = max(W, [], 2);
    newlabel = options(i);
end