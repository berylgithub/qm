% extracts relevant information given boundary matrix
function v = extractbound(bounds)
    pbu = 0; % var for (p)revious (b)ound (u)pperbound
    varsize = size(bounds)(2);
    v = zeros(varsize, 2); % output matrix
    for i=1:varsize
        % determine start index:
        if i==1
            v(i,1) = i;
        else
            v(i,1) = pbu + 1;
        end
        % determine end index:
        if bounds(3,i) == 2 % check for categorical
            v(i,2) = bounds(2,i) - bounds(1,i) + v(i,1);
        else
            v(i,2) = v(i,1);
        end
        pbu = v(i,2);
    end