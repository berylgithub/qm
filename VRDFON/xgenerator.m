

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% xgenerator.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to generate feasible x given (x,f)
function [x,xraw] = xgenerator(x, f, bounds)
    xraw=x=mintry(x,f); % the solver, this gives x raw
    for i = 1:length(x)
        tr = paramcheck(bounds(3,i), bounds(1,i), bounds(2,i)) % check whether need to be transformed or not
        if bounds(3,i) == 1 % check type, int = 1, real = 0
            x(i) = max(bounds(1,i), min(x(i), bounds(2,i))) % project to bounds
            % stochastic round:
            p = rand(1);
            xl = floor(x(i));
            f = x(i) - xl;
            if p < 1-f; % larger chance to be rounded to the closest int
                x(i) = xl;
            else
                x(i) = ceil(x(i));
            end
        else
            if tr
                lb = bounds(1,i)/(1+abs(bounds(1,i))); ub = bounds(2,i)/(1+abs(bounds(2,i))); % transform bounds
                bs = [lb, ub]; bs = sort(bs); % transform bounds
                z = max(bs(1), min(x(i), bs(2))) % clip to transformed bounds
                x(i) = z/(1-z)
            else
                x(i) = max(bounds(1,i), min(x(i), bounds(2,i))) % project to bounds
            end
        end
    end
