

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% paramtracker.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% wrapper to call xgenerator.m to generate new feasible x given (x,f,bounds)
% and checks whether (x,f) pair is already in (xlist, flist)
function [x, xraw, f, fpen, xlist, flist] = paramtracker(x, f, xlist, flist, bounds, bm)
    while true
        [x, xraw, fpen] = xgenerator(x, f, bounds, bm); % contains main loop to generate x given (x,f) and projection to feasible sol
        rid = find(ismember(xlist, x', "rows")) % check if x \in xlist
        if ~isempty(rid)
            f = flist(rid);
        else
            break
        end
    end