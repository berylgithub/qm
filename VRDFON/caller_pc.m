%par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
%par_fit_atom = [center_ids] # center_ids = 0 → use 
%par_fit = [model, cσ]
%params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
%           1     2        3           4           5               6             7       8     9
%naf, nmf in percentage, e.g., 50 -> .5*max(naf("ACSF"))
%feature name: 1=ACSF, 2=SOAP, 3=FCHL
%model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
%x = [.5, .5, 3, 1, 0, 0, 0, -6, -2048.0]' %  example
path_fun = '../data/hyperparamopt/fun.txt';
path_x = '../data/hyperparamopt/params.txt'
path_bounds = '../data/hyperparamopt/bounds.txt';
path_bounds_old = '../data/hyperparamopt/bounds_old.txt';
bounds = dlmread(path_bounds);
bounds_old = dlmread(path_bounds_old);

% x = [0.5; 0.5; 6; ]
xold = [0.5; 0.5; 5.5; 1/3; 1/3; 1/3; 1/2; 1/2; 1/2; 1/2; 38; 1/5; 1/5; 1/5; 1/5; 1/5; 10.5;]; % encoded x
f = 100.;

%x = [1/3; 1/3; 1/3; 5.5; 1/3; 1/3; 1/3; 1/2; 1/2; 1/2; 1/2; 1/5; 1/5; 1/5; 1/5; 1/5; 10.5]; 
%x = [1/3; 1/3; 1/3; 6; 1/3; 1/3; 1/3; 1/2; 1/2; 1/2; 1/2; 1/5; 1/5; 1/5; 1/5; 1/5; 11]% x mid for mixed int
%x = [1/3; 1/3; 1/3; 6; 1/2; 1/2; 1/2; 1/2; 1/2; 1/2; 1/5; 1/5; 1/5; 1/5; 1/5; 11]% x mid for mixed int without FCHL
%x = [0.306122; 0.081633; 0.612245; 6; 1; 0; 0; 1; 0; 1; 0; 0; 0; 0; 0; 1; 11]; % x best, MAE = 10.99
%x = [0.306122; 0.081633; 0.612245; 6; 0.6; 0.4; 0.4; 0.6; 0.4; 0.6; 4/25; 4/25; 4/25; 4/25; 4/25; 1/5; 11]; % current best found x with non deterministic round prob

%{
zi = [1/6, 1/3, 1/3, 1/6]'
xu = 20; xl = 1;
p = minprob(zi)
x = computex(p, xl, xu)
p = computep(x, xl, xu)
norm(p-zi)**2

bm = extractbound(bounds_old)
decode(xold, bounds_old, bm)

dlmread("../data/hyperparamopt/init_params.txt")

p = computep([16, 20], 1, 50)
x = computex([0.306122, 0.081633, 0.612245], 1, 50)

bounds
newbounds = boundtransform(bounds)
bm = extractbound(bounds);
[xnew, fpen] = decode(x, bounds, bm)
dlmwrite("../data/hyperparamopt/init_params.txt", x', "\t")

% test eigenvalue:
F = dlmread("../testeigen.txt");
C = corrcoef(F);
eig(C)
%}



%{
x = [0.306122; 0.081633; 0.612245; 6; 0.6; 0.4; 0.4; 0.6; 0.4; 0.6; 4/25; 4/25; 4/25; 4/25; 4/25; 1/5; 11]; % current best found x with non deterministic round prob
dlmwrite("../data/hyperparamopt/init_params.txt", x', "\t")

bounds
bm = extractbound(bounds);

xlist = []; # controller repo
for i=1:20 # loop over "processors"
    tol = 0
    while tol < 100 # tolerance loop to find next rounded iterates
        [xsim, fpen] = decode(x, bounds, bm)
        if isempty(find(ismember(xlist',xsim', "rows")))
            xlist = [xlist, xsim];
            break
        end
        tol += 1
    end
end

xlist


% finfo usage example:
path_fun = '../data/hyperparamopt/fun.txt';
t = 2; it_sim = [3,3,3]; f_sim = {{123., 1233., 101.},{[], 22., 102.},{[],13.,33}}; thres = 2;
t = finfo_updater(t, it_sim, f_sim, thres, path_fun)


% x_donator usage:
bounds
bm = extractbound(bounds);
xlist = [];
% loop over processors
for i = 1:20
    xlist = x_donator(x, xlist, bounds, bm)
end
xlist

% listener usage:
path_sim = "../data/hyperparamopt/sim/*.txt";
id_sim = []; f_id_sim = []; f_sim = {}; it_sim = []; cell_iter = 1;
i = 0
while true
    [id_sim, f_id_sim, f_sim, it_sim, cell_iter] = listener_sim(path_sim, id_sim, f_id_sim, f_sim, it_sim, cell_iter)
    i += 1
    pause(2)
end

%}

f = {"../data/hyperparamopt/sim/f/sim_11.txt";"../data/hyperparamopt/sim/f/sim_2.txt"; "../data/hyperparamopt/sim/f/sim_3.txt"};
%{
for i=1:length(f)
    delete(f{i})
end
%}

% artificial f info from simulator for testing, [state, iter, uid, f]
%{
dlmwrite(f{1}, [1, 2, 0.21, 2.], "\t")
dlmwrite(f{2}, [1, 2, 0.22, 2.2], "\t")
dlmwrite(f{3}, [1, 2, 0.23, 2.3], "\t")
%}

% for resetting file trackers:
%{
path_trackx = '../data/hyperparamopt/xlist.txt'; 
path_trackxraw = '../data/hyperparamopt/xrawlist.txt';
path_trackf = '../data/hyperparamopt/flist.txt';
path_simtracker = "../data/hyperparamopt/sim/sim_tracker.txt";
tlist = {path_trackx, path_trackxraw, path_trackf, path_simtracker};
for t = 1:length(tlist)
    dlmwrite(tlist{t}, []) % write empty files
end
%}


%p = computep([16, 20], 1, 50)
%x = computex([0.306122, 0.081633, 0.612245], 1, 50)

%  =============================================
% (15.09.23) hyperparam for the current best (7.59 kcal/mol) = [ACSF_51, REAPER, only DB = true, D_PCAs = falses, F_PCAs = falses, normalizes = falses]:
% see expdriver for the hyperparameter ordering
% xd = [1, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 1, 0, 0, 6, 11] # the decoded x
% the encoded x (the one that needs to be passed to solver):
x = [   1;0; 
        1;0;
        1;0;
        1;0;
        1;0;
        1;0;
        10;
        10;
        10;
        1;0;
        1;0;
        1;0;0;
        3;
        0;0;0;0;1;
        1;0;
        1;0;
        0;0;0;0;1;0;
        11;
        0;1
    ]
y = [rand(1); x]
% generate bounds:
bounds = [  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1;
            1, 1, 1, 1, 1, 1, 10, 10, 10, 1, 1, 50, 10, 5, 1, 1, 6, 20, 2;
            2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2, 2, 1, 2;
            2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 3, 1, 5, 2, 2, 6, 1, 2]
bm = extractbound(bounds) % compute boundary index matrix
[xout, fpen] = decode(x, bounds, bm) % check if the decoded x is correct
dlmwrite("../data/hyperparamopt/init_params.txt", [rand(1); x]', "\t")
dlmwrite("../data/hyperparamopt/bounds.txt", bounds, "\t")