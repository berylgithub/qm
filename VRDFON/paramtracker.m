% appender of (x,f) to (xlist, flist)
function [x, f, xlist, flist] = paramtracker(x, f, xlist, flist)
    while true
        x = xgenerator(x, f); % contains main loop to generate x given (x,f) and projection to feasible sol
        rid = find(ismember(xlist, x', "rows")) # check if x \in xlist
        if ~isempty(rid)
            f = flist(rid);
        else
            break
        end
    end