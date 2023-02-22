% project infeasible sol to the bounds of feasible sol
function x=parambound(x, bounds)
    % first (naf, nmf) in real:
    xtest = max(xlo, min(xtest, xhi)) % clip to boundary



