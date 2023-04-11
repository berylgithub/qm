

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% computep.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computes "probability vector" p given ordered integral vector x
function p = computep(x, xl, xu)
    dx = xu - xl;
    n = length(x);
    p = zeros(n+1, 1);
    for i=1:n+1
        if i == 1 % initial condition
            p(i) = (x(i) - xl)/dx;
        elseif i == n+1 % boundary condition
            p(i) = (xu - x(i-1))/dx;
        else
            p(i) = (x(i) - x(i-1))/dx;
        end
    end
