function xlist = x_donator(x, xlist, bounds, bm, xpath, iter)
    tol = 0;
    xsim = 0; % init empty x
    while tol < 100 %tolerance loop to find next rounded iterates
        [xsim, fpen] = decode(x, bounds, bm);
        if isempty(find(ismember(xlist',xsim', "rows"))) % if xsim is found in the list, then break and write the xsim to the corresponding simulator
            xlist = [xlist, xsim];
            break
        end
        tol += 1;
    end
    dlmwrite(xpath, [iter, xsim'], "\t");