

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% paramtracker.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% wrapper to call xgenerator to generate new feasible x given (x,f,bounds)
% and checks whether (x,f) pair is already with (xlist, flist)
function [x, xraw, f, xlist, flist] = paramtracker(x, f, xlist, flist, bounds)
    while true
        [x, xraw] = xgenerator(x, f, bounds); % contains main loop to generate x given (x,f) and projection to feasible sol
        rid = find(ismember(xlist, x', "rows")) % check if x \in xlist
        if ~isempty(rid)
            f = flist(rid);
        else
            break
        end
    end