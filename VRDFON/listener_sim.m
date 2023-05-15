% listens to simulators: checking whether the simulators give any update
% give x to simulators: happens if fid is updated, or sim entry

function [id_sim, f_id_sim, f_sim, it_sim, cell_iter] = listener_sim(path_sim, id_sim, f_id_sim, f_sim, it_sim, cell_iter)
    files = dir(path_sim);
    if !isempty(files)
        % check whether there is new simulator:
        % get simulator ids:
        for i=1:length(files)
            fname = files(i).name; % filename of the output of the simulator i
            id = strsplit(fname, "_")(2); % the id is 2nd entry
            finder = find(ismember(id_sim, id)); % find by id
            if isempty(finder) % (sim entry) if id not found, initialize simulator info:
                id_sim = [id_sim; id]; % append new sim id
                sinfo = dlmread(strcat(path_sim(1:end-5),fname)); % get sim info
                if length(sinfo) == 0 % if sim is empty then init iter with 0
                    it_sim = [it_sim; 0]; % init iter
                    f_id_sim = [f_id_sim; 0]; % alloc id with 0, f_id should be float from rand(1)                
                    f_sim{cell_iter} = []; % init empty f value from simulator
                    cell_iter += 1; % increment cell iter tracker
                    % give x (write x in specific file by sim id):
                    % rounding procedure wrapper
                % else, add sinfo (add later, optional):
                end
            else % if id is found, get updated info from sim[id]:
                sinfo = dlmread(strcat(path_sim(1:end-5),fname)) % get sim data
                if length(sinfo) > 0 % if not empty then the sim has ever computed f value
                    f_id = sinfo(1); % check if the f id is different:
                    if f_id != f_id_sim(finder) % (fid update) if it's different, then new f has been computed (f, it, f_id)
                        f_sim{finder} = [f_sim{finder}; sinfo(2)]; % update f
                        it_sim(finder) += 1; % increment iteration
                        f_id_sim(finder) = f_id; % update f_id tracker
                        % give x:
                    end
                end
            end
        end
    end
