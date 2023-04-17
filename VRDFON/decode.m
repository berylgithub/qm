
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% decode.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to decode encoded x, 
% given vector "x", bounds matrix "bounds", and bounds index matrix "bm", 
% this generally implies reduction of x's dimension.
% Also computes penalty objective but is independent of function value!!
function [xnew, fpen] = decode(x, bounds, bm)
    xnew = []; % new x to be accepted by Julia
    fpen = 0; % f penalty
    bsize = size(bounds)(2); % number of variables
    for i = 1:bsize
        % check whether need to be transformed or not:
        tr = paramcheck(bounds(3,i), bounds(1,i), bounds(2,i));
        % slice x_i given the index matrix bm
        zi = x(bm(i,1):bm(i,2)); % zi := encoded xi, i.e., raw xi
        % check type, real = 0, int = 1, categorical = 2, ordered int =3
        if bounds(3,i) == 1 % int
            % project to bounds:
            xi = max(bounds(1,i), min(zi, bounds(2,i)));
            % stochastic round:
            r = rand(1);
            xl = floor(xi);
            p = xi - xl;
            if r < 1-p; % larger chance to be rounded to the closest int
                xi = xl;
            else
                xi = ceil(xi);
            end
            fpen += abs(xi - zi)/bounds(2,i);
        elseif bounds(3,i) == 2 % cat
            p = minprob(zi); % get probability from solving some min problem
            % choose x_i based on p:
            xi = 0;
            q = [0.; cumsum(p)];
            r = rand(1);
            v = linspace(bounds(1,i), bounds(2,i), bounds(2,i)-bounds(1,i)+1); % generate the category vector
            p = 0*p; % reset p to zeros for fpen
            for j=1:length(q)-1
                if (q(j) < r) && (r <= q(j+1))
                    xi = v(j);
                    p(j) = 1;
                    break
                end
            end
            fpen += norm(p-zi,1);  % compute penalty
        elseif bounds(3,i) == 3 % oint
            p = minprob(zi);
            xi = computex(p, bounds(1,i), bounds(2,i));
            p = computep(xi, bounds(1,i), bounds(2,i));
            fpen += norm(p-zi)**2;  % compute penalty
        else
            if tr
                lb = bounds(1,i)/(1+abs(bounds(1,i))); 
                ub = bounds(2,i)/(1+abs(bounds(2,i))); % transform bounds
                bs = [lb, ub]; bs = sort(bs); % transform bounds
                z = max(bs(1), min(zi, bs(2))); % clip to transformed bounds
                xi = z/(1-z);
            else
                % project to bounds:
                xi = max(bounds(1,i), min(zi, bounds(2,i)));
            end
            fpen += abs(xi - zi)/bounds(2,i);
        end
        xnew = [xnew; xi];
    end

