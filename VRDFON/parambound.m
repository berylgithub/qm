% project infeasible sol to the bounds of feasible sol
function x=parambound(x, bounds)
    for i=1:length(x)
        x(i) = max(bounds(1,i), min(x(i), bounds(2,i))); % clip to boundary
    end



