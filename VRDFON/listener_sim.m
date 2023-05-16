% listens to simulators: checking whether the simulators give any update
% sinfo is simulator info, contains: id_sim (from file name), state, iter of sim, f
% give x to simulators: happens if state is idle

function [id_sim, f_sim, it_sim, cell_iter] = listener_sim(path_simf, path_simx, id_sim, f_sim, it_sim, cell_iter, iter_tracker)
    files = dir(path_simf);
    if !isempty(files)
        % check whether there is new simulator:
        % get simulator ids:
        for i=1:length(files)
            fname = files(i).name; % filename of the output of the simulator i
            id = strsplit(fname, "_")(2); % the id is 2nd entry
            finder = find(ismember(id_sim, id)); % find by id
            sinfo = dlmread(strcat(path_simf(1:end-5),fname)); % get sim info
            if isempty(finder) % (sim entry) if id not found, initialize simulator info:
                % initialize sinfo:
                id_sim = [id_sim; id]; % append new sim id
                it_sim = [it_sim; 0]; % init iter
                f_sim{cell_iter} = {}; % init empty f value from simulator
                if length(sinfo) > 2 % if length(sinfo) == 3 then the sim has ever computed something
                    sl_iter = sinfo(2); % the last iteration of the observed simulator 
                    sl_f = sinfo(3); % last fobj of the observed simulator
                    f_sim{cell_iter}{sl_iter} = sl_f; % add f and iter id
                end
                cell_iter += 1; % increment cell iter tracker
                % give x if sim is idle (0):
                if !isempty(sinfo)
                    if sinfo(1) == 0
                        disp("x has been given!")
                        str = strcat(path_simx, fname)
                        %dlmwrite(str, [iter_tracker+1, ])
                    end
                end

            else % if id is found, get updated info from sim[id]:
                if length(sinfo) > 2
                    % get f:
                    sl_state = sinfo(1); % sim state
                    sl_iter = sinfo(2); % sim iteration
                    sl_f = sinfo(3); % sim fobj
                    f_sim{finder}{sl_iter} = sl_f; % set f in the cell
                    it_sim(finder) = sl_iter;
                    % give x if sim is idle (0):
                    if sl_state == 0
                        disp("x has been given")
                    end
                end
            end
        end
    end
