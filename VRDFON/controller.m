% controls the flow between simulators and mintry:
%   - init controller info: 
%       - number of simulators (allocate)
%       - init filepaths
%   - periodically:
%       - load new x from mintry whenever it is updated
%       - checks for new signal from simulators:
%           - takes new f if the simulator places it, if the current iteration quota is fulfilled, return the best (rounded x, f) to mintry
%           - give rounded x to simulators that sends request
%       
%       

path_simfolder = "../data/hyperparamopt/sim/"
path_siminfo = "../data/hyperparamopt/sim_info.txt"; % init num sim from here
path_bounds = '../data/hyperparamopt/bounds.txt'; % bound info for rounding init
path_rawparam = '../data/hyperparamopt/raw_params.txt'; % get xraw from here
path_fun = '../data/hyperparamopt/fun.txt'; % store best f here

disp("init controller...")
% init sim info
siminfo = dlmread(path_siminfo);
%n_sim = siminfo(1); % number of spawned simulators, probably not used later, since the number of sim will be dynamic instead
t_sim = siminfo(1); % integral threshold of when each iteration is fulfilled, need to know beforehand the expected number of simulator will be spawned, otherwise each iteration will never be finished
id_sim = []; f_sim = []; it_sim = []; % initialization of simulator trackers

% extract boundary info
bounds = dlmread(path_bounds);
bm = extractbound(bounds);

i = 0 % remove later
rp_id = 99999999; xraw = []; % dummy xraw data
while true
    % listens to xraw port:
    rp_info = dlmread(path_rawparam);
    if rp_info(1) != rp_id
        disp("new incoming data!")
        rp_id = rp_info(1);
        xraw = rp_info(2:end);
    end
    pause(.5)
    % listens to simulator port
    [id_sim, f_sim, it_sim] = listener_sim(path_sim, id_sim, f_sim, it_sim) 
    % computes argmin_i f(i) and store f(i) in path_fun:
    i += 1 % remove later
end
