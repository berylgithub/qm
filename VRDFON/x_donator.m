function xlist = x_donator(x, xlist, bounds, bm)
    tol = 0;
    while tol < 100 %tolerance loop to find next rounded iterates
        [xsim, fpen] = decode(x, bounds, bm);
        if isempty(find(ismember(xlist',xsim', "rows")))
            xlist = [xlist, xsim];
            break
        end
        tol += 1
    end