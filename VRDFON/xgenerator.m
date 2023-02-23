function x=xgenerator(x, f)
    feas=false;
    %stuckcount=1;
    while feas==false
        disp(feas)
        disp(x)
        disp(f)
        x=mintry(x,f); % the solver
        % round x by probablity:
        for i = 3:length(x)
            p = rand(1);
            xl = floor(x(i));
            f = x(i) - xl
            if p < 1-f; % larger chance to be rounded to the closest int
                x(i) = xl;
            else
                x(i) = ceil(x(i));
            end
        end
        feas=paramcheck(x) % replace with projection
        if !feas
            f = 100.; % set supremum MAE
        end
    end