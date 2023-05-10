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

path_siminfo = "../data/hyperparamopt/sim_info.txt"; % init num sim from here
path_bounds = '../data/hyperparamopt/bounds.txt'; % bound info for rounding init
path_rawparam = '../data/hyperparamopt/raw_params.txt'; % get xraw from here
path_fun = '../data/hyperparamopt/fun.txt'; % store best f here

disp("init controller...")
% extract boundary info
bounds = dlmread(path_bounds)
bm = extractbound(bounds)
n_sim = dlmread(path_siminfo)