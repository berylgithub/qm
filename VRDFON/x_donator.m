function xlist = x_donator(x, xlist, bounds, bm, xpath, iter)
    tol = 0;
    xsim = []; % init empty x
    while tol < 100 %tolerance loop to find next rounded iterates
        [xsim, fpen] = decode(x, bounds, bm);
        if isempty(find(ismember(xlist',xsim', "rows"))) % if xsim is found in the list, then break and write the xsim to the corresponding simulator
            xlist = [xlist, xsim];
            break
        end
        tol += 1;
    end
    disp(strcat("x given to ", xpath));
    disp(xsim);
    if !isempty(xsim) % means the x is decoded atleast once
        dlmwrite(xpath, [rand(1), iter, xsim'], "\t");
    end