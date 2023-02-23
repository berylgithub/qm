% appender of (x,f) to (xlist, flist)
function [x, f, xlist, flist, rid] = paramtracker(x, f, xlist, flist)
    rid = find(ismember(xlist, x, 'rows')) # check if x \in xlist
    if isempty(rid)
        xlist = [xlist; x]; flist = [flist f];
    else
        x = xlist(rid, :); f = flist(rid);
    end