

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% xgenerator.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to generate feasible x given (x,f)
function [x,xraw] = xgenerator(x, f, bounds)
    xraw=x=mintry(x,f); % the solver, this gives x raw
    x = min(1., max(x, 0.)); % now all variables are (0, 1), the translation is handled by Julia
    % this one is mixed variable types mode, larger search space
    %{
    for i = 3:length(x)
        p = rand(1);
        xl = floor(x(i));
        f = x(i) - xl;
        if p < 1-f; % larger chance to be rounded to the closest int
            x(i) = xl;
        else
            x(i) = ceil(x(i));
        end
    end
    x=parambound(x, bounds); % project to bounds, feasible guaranteed, this is processed x

    }%
    
    % this one is "check for feasibility" version instead of projection: 
    %{
    feas=false;
    while feas==false
        disp("infeasible!")
        x=mintry(x,f); % the solver
        % round x by probablity:
        for i = 3:length(x)
            p = rand(1);
            xl = floor(x(i));
            f = x(i) - xl;
            if p < 1-f; % larger chance to be rounded to the closest int
                x(i) = xl;
            else
                x(i) = ceil(x(i));
            end
        end
        feas=paramcheck(x); % replace with projection
        if !feas
            f = 100.; % set supremum MAE
        end
    end
    %}