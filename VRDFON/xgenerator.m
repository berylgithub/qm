

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% xgenerator.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to compute new x (in encoded format), given (x,f,bounds).
% also computes f penalty.
% the indexing is able to accomodate encoded x, using the bm matrix.
% calls mintry to compute new x in encoded format
% calls decoded to decode x
function [x, xraw, fpen] = xgenerator(xraw, f, bounds, bm)
    xraw = mintry(xraw,f); % the solver, this gives x raw
    [x, fpen] = decode(xraw, bounds, bm); % decode the new x to be sent to Julia, and computes fpen
