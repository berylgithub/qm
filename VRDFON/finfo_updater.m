% if count(it_sim > iter_tracker) > threshold, then get best(f(trues)[tracker+1])
% then store in "fun.txt" tracker file as usual
function iter_tracker = finfo_updater(iter_tracker, it_sim, f_sim, thres, path_fun)
    finder = it_sim > iter_tracker
    if sum(finder) >= thres
        % find fbest:
        fbest = Inf;
        for i=1:length(finder)
            if finder(i) == 1
                fsel = f_sim{i}(iter_tracker+1);
                if fsel < fbest
                    fbest = fsel;
                end
            end
        end
        iter_tracker +=1;
        % store fbest in fun,txt:
        dlmwrite(path_fun, [rand(1), fbest], "\t");
    end
