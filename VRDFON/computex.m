

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% computex.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function which computes integral x given a "probability vector" p \in [0,1]
function x = computex(p, xl, xu)
    dx = xu - xl;
    n = length(p);
    x = zeros(n-1, 1);
    for i=1:n-1
        if i == 1 % init cond
            x(i) = xl + p(i)*dx;
        else
            x(i) = x(i-1) + p(i)*dx;
        end
    end
    x = round(x);
