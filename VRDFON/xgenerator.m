

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% xgenerator.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to generate feasible x given (x,f)
function [x,xraw] = xgenerator(x, f, bounds)
    xraw=x=mintry(x,f); % the solver, this gives x raw
    x=parambound(x, bounds); % project to bounds, feasible guaranteed
    for i = 1:length(x)
        % project to boundary and round here:
        p = rand(1);
        xl = floor(x(i));
        f = x(i) - xl;
        if p < 1-f; % larger chance to be rounded to the closest int
            x(i) = xl;
        else
            x(i) = ceil(x(i));
        end
    end
    %x = min(1., max(x, 0.)); % now all variables are (0, 1), the translation is handled by Julia