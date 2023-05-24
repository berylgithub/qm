% listens to simulators: checking whether the simulators give any update
% sinfo is simulator info, contains: id_sim (from file name), state, iter of sim, fid, f
% give x to simulators: happens if state is idle

function [id_sim, f_sim, fid_sim, xlist] = listener_sim(path_simf, id_sim, f_sim, fid_sim, iter_tracker, xd_vars)
    files = dir(path_simf);
    % unroll params of xdonator:
    xraw = xd_vars{1}; xlist = xd_vars{2}; bounds = xd_vars{3}; bm = xd_vars{4}; path_simx = xd_vars{5};
    if !isempty(files)
        % check whether there is new simulator:
        % get simulator ids:
        for i=1:length(files)
            fname = files(i).name; % filename of the output of the simulator i
            id = str2num(strsplit(strsplit(fname, "_"){2}, "."){1}); % the sim id is the integral str before the dot after underscore
            finder = find(ismember(id_sim, id)); % find by id
            sinfo = dlmread(strcat(path_simf(1:end-5),fname)); % get sim info
            if isempty(finder) % (sim entry) if id not found, initialize simulator info:
                % initialize sinfo:
                id_sim = [id_sim; id]; % append new sim id
                fid_sim{id} = 0. ; % init fid, must be float
                if length(sinfo) > 2 % if length(sinfo) == 3 then the sim has ever computed something
                    sl_iter = sinfo(2); % the last iteration of the observed simulator 
                    sl_fid = sinfo(3); % the id of sim comp 
                    sl_f = sinfo(4); % last fobj of the observed simulator
                    fid_sim{id} = sl_fid; % set fid
                    %f_sim{iter_tracker} = [f_sim{iter_tracker}, sl_f]; % add f and iter id. (?): probably logic incorrect, should init fsim[simiter] if it's empty, otherwise add to it.
                    %s probably should be this instead (NOT YET TESTED):
                    if isempty(f_sim{sl_iter})
                        f_sim{sl_iter} = [];
                    else
                        f_sim{sl_iter} = [f_sim{sl_iter}, sl_f];
                    end
                end
                % give x if sim is idle (0):
                if !isempty(sinfo)
                    if sinfo(1) == 0
                        %disp("new simulator, give x")
                        xlist = x_donator(xraw, xlist, bounds, bm, strcat(path_simx, fname), iter_tracker)
                    end
                end
            else % if id is found, get updated info from sim[id]:
                if length(sinfo) > 2
                    % get f:
                    sl_state = sinfo(1); % sim state
                    sl_iter = sinfo(2); % sim iteration
                    sl_fid = sinfo(3); % the fid of the comp from the sim
                    sl_f = sinfo(4); % sim fobj
                    % it the fid is different than the previous one, then update f:
                    if fid_sim{id} != sl_fid
                        f_sim{sl_iter} = [f_sim{sl_iter}, sl_f]; % set f in the cell
                        fid_sim{id} = sl_fid;
                    end
                    % give x if sim is idle (0):
                    if sl_state == 0
                        %disp("old simulator, idle state, give x")
                        xlist = x_donator(xraw, xlist, bounds, bm, strcat(path_simx, fname), iter_tracker)
                    end
                end
            end
        end
    end
