

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% xgenerator.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to generate feasible x given (x,f)
function [x,xraw] = xgenerator(x, f, bounds)
    xraw=x=mintry(x,f); % the solver, this gives x raw
    x = min(1., max(x, 0.)); % now all variables are (0, 1), the translation is handled by Julia