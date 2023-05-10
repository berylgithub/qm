function [id_sim, f_sim, it_sim] = listener_sim(path_sim, id_sim, f_sim, it_sim)
    files = dir(path_sim);
    if !isempty(files)
        % check whether there is new simulator:
        % get simulator ids:
        for i=1:length(files)
            id = strsplit(files(i).name, "_")(2);
            finder = find(ismember()) % find by id
            if isempty(finder)
                id_sim = [id_sim; id]; % append new sim id
                % append the f
                % start iter
            else
                
            end
        end
    end
