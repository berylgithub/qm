% if count(it_sim > iter_tracker) > threshold, then get best(f(trues)[tracker+1])
% then store in "fun.txt" tracker file as usual
function iter_tracker = finfo_updater(iter_tracker, f_sim, thres, path_fun)
    fs = f_sim{iter_tracker};
    lenf = length(fs);
    if lenf >= thres
        % find fbest:
        fbest = Inf;
        for i=1:lenf
            fsel = fs(i);
            if fsel < fbest
                fbest = fsel;
            end
        end
        fbest
        iter_tracker += 1;
        % store fbest in fun,txt:
        dlmwrite(path_fun, [rand(1), fbest], "\t");
    end