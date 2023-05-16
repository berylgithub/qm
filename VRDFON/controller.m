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
path_simf = strcat(path_simfolder, "f/"); % contains fobj info which are the output of the simulators 
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
iter_tracker = 0 % tracks the internal iteration in which it determines how many iters have the the threshold been fulfied

% init simulator listener:
path_sim = strcat(path_simfolder, "*.txt");
id_sim = []; % a vector, the id of simulators
f_sim = {};  % a cell of cells, each cell = each simulator, each cell element = fobj of the corresponding iteration
it_sim = []; % a vector, determines on which iteration each simulator is in 
cell_iter = 1; % int, keeps track of the occupied cells of f_sim, to avoid replacing the celss whenever new sim enters

% init x donator controller
xlist = []; % contains only the list of possible rounding iterates in each mintry iteration

% extract boundary info
bounds = dlmread(path_bounds);
bm = extractbound(bounds);

i = 0 % remove later
rp_id = 99999999; xraw = []; % init xraw data
while true
    % listens to xraw port:
    rp_info = dlmread(path_rawparam);
    if rp_info(1) != rp_id
        disp("new incoming data!")
        rp_id = rp_info(1);
        xraw = rp_info(2:end);
    end
    % checks for simulator's global counter update, must be BEFORE any simulator data update:
    iter_tracker = finfo_updater(iter_tracker, it_sim, f_sim, thres, path_fun)
    % listens to simulator port, and updates simulator data:
    [id_sim, f_sim, it_sim, cell_iter] = listener_sim(path_simf, id_sim, f_sim, it_sim, cell_iter, iter_tracker)
    i += 1 % remove later
    pause(2)
end
