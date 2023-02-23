% function to generate feasible x given (x,f)
function x=xgenerator(x, f, bounds)
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
    x=parambound(x, bounds) % project to bounds, feasible guaranteed

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