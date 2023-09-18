% controls the flow between simulators and mintry:
%   - init controller info: 
%       - number of simulators (allocate)
%       - init filepaths
%   - periodically:
%       - load new x from mintry whenever it is updated
%       - checks whether the current iter quota is fulfilled given the threshold, if yes then write the best f for mintry
%       - checks for new signal from simulators:
%           - takes new f if the simulator places it
%           - give rounded x to simulators that sends request

path_simfolder = "../data/hyperparamopt/sim/"; % contains all data of simulators
path_simx = strcat(path_simfolder, "x/"); % contains x info of which given by controller to simulators
path_simf = strcat(path_simfolder, "f/*.txt"); % contains fobj info which are the output of the simulators 
path_siminfo = "../data/hyperparamopt/sim/sim_info.txt"; % init num sim from here
path_bounds = '../data/hyperparamopt/bounds.txt'; % bound info for rounding init
path_rawparam = '../data/hyperparamopt/raw_params.txt'; % get xraw from here
path_fun = '../data/hyperparamopt/fun.txt'; % store best f here

disp("init controller...")
% init sim info
siminfo = dlmread(path_siminfo);
%n_sim = siminfo(1); % number of spawned simulators, probably not used later, since the number of sim will be dynamic instead
thres = siminfo(1); % integral threshold of when each iteration is fulfilled, need to know beforehand the expected number of simulator will be spawned, otherwise each iteration will never be finished

% init fbest controller:
prev_iter = iter_tracker = 1 % tracks the internal iteration in which it determines how many iters have the the threshold been fulfied

% init simulator listener:
path_sim = strcat(path_simfolder, "*.txt");
id_sim = []; % a vector, the id of simulators
f_sim = {};  % a cell of vectors, each cell = each iteration, each cell element = fobj of the corresponding iteration
f_sim{iter_tracker} = []; % init empty vector for the current iter
fid_sim = {}; % tracker of fobj comp id, to make sure no duplicate computation result is inputted

% extract boundary info
bounds = dlmread(path_bounds)
bm = extractbound(bounds);

% init x donator controller
xd_vars{1} = []; xd_vars{2} = []; xd_vars{3} = bounds; xd_vars{4} = bm; xd_vars{5} = path_simx; % container of x donator variables
xlist = []; % contains only the list of possible rounding iterates for each mintry iteration, reset after each mintry iter is finished

i = 0 % remove later
rp_id = 99999999; xraw = []; % init xraw data
while true
    % listens to xraw port:
    if dir(path_rawparam).bytes > 0
        rp_info = dlmread(path_rawparam);
        if rp_info(1) != rp_id
            disp("new incoming xraw!")
            rp_id = rp_info(1); % uid of the raw x
            xraw = rp_info(2:end)' % transpose to column vector
        end
    end
    % if iter_tracker is changed (incremented):
    if iter_tracker != prev_iter
        xlist = [] % reset xlist
        f_sim{prev_iter} = [] % empty previous cell, memory reason
        if numel(f_sim) < iter_tracker
            f_sim{iter_tracker} = [] % initialize next iter's cell if it's not yet there
        end
        prev_iter = iter_tracker % reset prev iter since iter tracker is changed
    end
    %xlist
    % gives new f to mintry if the number of new iterates > thres (see the function logic), must be BEFORE any simulator data update:
    iter_tracker = finfo_updater(iter_tracker, f_sim, thres, path_fun);
    % listens to simulator port, and updates simulator data:
    if !isempty(xraw)
        xd_vars{1} = xraw; xd_vars{2} = xlist; % fill the vars for x_donator fun
        [id_sim, f_sim, fid_sim, xlist] = listener_sim(path_simf, id_sim, f_sim, fid_sim, iter_tracker, xd_vars);
        id_sim
        f_sim
        fid_sim
    end
    %i += 1 % remove later
    pause(2) % for easier debugging, increase speed later
end
