% appender of (x,f) to (xlist, flist)
function [f, xlist, flist] = paramtracker(x, f, xlist, flist)
    rid = find(ismember(xlist, x', "rows")) # check if x \in xlist
    if isempty(rid)
        xlist = [xlist; x']; flist = [flist f];
    else
        f = flist(rid);
    end