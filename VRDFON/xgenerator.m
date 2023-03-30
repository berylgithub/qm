

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% xgenerator.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to generate feasible x (and xraw for tracking) given (x,f,bounds)
% the indexing need is able to accomodate encoded x, uusing the bm matrix
function [x,xraw] = xgenerator(x, f, bounds, bm)
    xraw=x=mintry(x,f); % the solver, this gives x raw
    xnew = []; % new x to be accepted by Julia
    bsize = size(bounds)(2) % number of variables
    for i = 1:bsize
        % check whether need to be transformed or not:
        tr = paramcheck(bounds(3,i), bounds(1,i), bounds(2,i));
        % check type, int = 1, real = 0, categorical = 2
        xi = x(bm(i,1):bm(i,2)); % slice x_i given the index matrix bm
        if bounds(3,i) == 1 
            % project to bounds:
            xi = max(bounds(1,i), min(xi, bounds(2,i)));
            % stochastic round:
            p = rand(1);
            xl = floor(xi);
            f = xi - xl;
            if p < 1-f; % larger chance to be rounded to the closest int
                xi = xl;
            else
                xi = ceil(xi);
            end
        elseif bounds(3,i) == 2
            p = minprob(xi); % get probability from solving some min problem
            % choose x_i based on p:
            xi = 0;
            q = [0.; cumsum(p)];
            for i=1:length(q)-1
                if (q(i) < r) && (r <= q(i+1))
                    xi = i;
                    break
                end
            end
        else
            if tr
                lb = bounds(1,i)/(1+abs(bounds(1,i))); 
                ub = bounds(2,i)/(1+abs(bounds(2,i))); % transform bounds
                bs = [lb, ub]; bs = sort(bs); % transform bounds
                z = max(bs(1), min(xi, bs(2))); % clip to transformed bounds
                xi = z/(1-z);
            else
                % project to bounds:
                xi = max(bounds(1,i), min(xi, bounds(2,i)));
            end
        end
    end
