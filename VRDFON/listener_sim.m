% listens to simulators: checking whether the simulators give any update
% sinfo is simulator info, contains: id_sim (from file name), state, iter of sim, f
% give x to simulators: happens if state is idle

function [id_sim, f_id_sim, f_sim, it_sim, cell_iter] = listener_sim(path_sim, id_sim, f_id_sim, f_sim, it_sim, cell_iter, iter_tracker)
    files = dir(path_sim);
    if !isempty(files)
        % check whether there is new simulator:
        % get simulator ids:
        for i=1:length(files)
            fname = files(i).name; % filename of the output of the simulator i
            id = strsplit(fname, "_")(2); % the id is 2nd entry
            finder = find(ismember(id_sim, id)); % find by id
            sinfo = dlmread(strcat(path_sim(1:end-5),fname)); % get sim info
            if isempty(finder) % (sim entry) if id not found, initialize simulator info:
                % initialize sinfo:
                id_sim = [id_sim; id]; % append new sim id
                it_sim = [it_sim; 0]; % init iter
                f_sim{cell_iter} = {}; % init empty f value from simulator
                if length(sinfo) > 2 % if length(sinfo) == 3 then the sim has ever computed something
                    sl_iter = sinfo(2); % the last iteration of the observed simulator 
                    sl_f = sinfo(3); % last f of the observed simulator
                    f_sim{cell_iter}{sl_iter} = sl_f; % add f and iter id
                end
                cell_iter += 1; % increment cell iter tracker
                
                % give x if sim is idle:

            else % if id is found, get updated info from sim[id]:
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
