using Hyperopt, NOMAD

include("utils.jl")
include("alouEt.jl")

function caller_ds()
#=     # FEEATURE EXTRACTION:
    foldername = "exp_reduced_energy"
    # ACSF:
    nafs = [40, 30, 20] 
    nmfs = [40, 30, 20]
    println("ACSF")
    for i ∈ eachindex(nafs)
        for j ∈ eachindex(nmfs)
            nmf = nmfs[j]
            if nmf >= nafs[i] # the num of mol feature must be less than atom f
                nmf = nafs[i] - rand(1:5, 1)[1]
            end
            println(nafs[i]," ",nmf)
            data_setup(foldername, nafs[i], nmf, 3, 300, "data/qm9_dataset_old.jld", "data/ACSF.jld", "ACSF"; 
                        save_global_centers = true, num_center_sets = 5)
            GC.gc()
        end
    end
    # SOAP:
    nafs = [75, 50, 25]
    nmfs = [75, 50, 25]
    println("SOAP")
    for i ∈ eachindex(nafs)
        for j ∈ eachindex(nmfs)
            nmf = nmfs[j]
            if nmf >= nafs[i] # the num of mol feature must be less than atom f
                nmf = nafs[i] - rand(1:5, 1)[1]
            end
            println(nafs[i]," ",nmf)
            data_setup(foldername, nafs[i], nmf, 3, 300, "data/qm9_dataset_old.jld", "data/SOAP.jld", "SOAP"; 
                        save_global_centers = true, num_center_sets = 5)
            GC.gc()
        end
    end =#

    # ATOM FITTING:
    uids = readdlm("data/centers.txt")[:, 1]
    kids = readdlm("data/centers.txt")[:, 2]
    center_sets = Int.(readdlm("data/centers.txt")[:, 3:end]) # a matrix (n_sets, n_centers) ∈ Int
    display(center_sets)
    for i ∈ eachindex(uids)
        println(uids[i]," ",kids[i])
        fit_atom("exp_reduced_energy", "data/qm9_dataset_old.jld", "data/atomref_features.jld";
              center_ids=center_sets[i, :], uid=uids[i], kid=kids[i])
        GC.gc()
    end
end


function caller_fit()
    # get the 10 best:
    atom_info = readdlm("result/exp_reduced_energy/atomref_info.txt")[99:193, :] # the first 3 is excluded since theyr noise from prev exps
    MAVs = Vector{Float64}(atom_info[:, 3]) # 3:=training MAE, 4:=training+testing MAE
    sid = sortperm(MAVs) # indices from sorted MAV
    #best10 = atom_info[sid[1:10], :] # 10 best
    best10 = atom_info[sid, :]
    display(best10)
    # compute total atomic energy:
    E_atom = Matrix{Float64}(best10[:, end-4:end]) # 5 length vector
    E = vec(readdlm("data/energies.txt"))
    f_atomref = load("data/atomref_features.jld", "data") # n × 5 matrix
    ds = readdlm("data/exp_reduced_energy/setup_info.txt")
    ds_uids = ds[:, 1] # vector of uids
    centers_info = readdlm("data/centers.txt")
    centers_uids = centers_info[:, 1]
    centers_kids = centers_info[:, 2]
    models = ["ROSEMI", "KRR", "NN", "LLS", "GAK"] # for model loop, |ds| × |models| = 50 exps
    for k ∈ axes(best10, 1) # loop k foreach found best MAV
        E_null = f_atomref*E_atom[k,:] # Ax, where A := matrix of number of atoms, x = atomic energies
        #MAV = mean(abs.(E - E_null))*627.5
        id_look = best10[k, 1] # the uid that we want to find
        id_k_look = best10[k, 2] # the uid of the center that we want to find
        # look for the correct centers:
        found_ids = []
        for i ∈ eachindex(centers_uids)
            if id_look == centers_uids[i]
                push!(found_ids, i)
            end 
        end
        #display(centers_info[found_ids, :])
        # get the correct center id:
        found_idx = nothing
        for i ∈ eachindex(found_ids)
            if id_k_look == centers_kids[found_ids[i]]
                found_idx = i
                break
            end
        end
        #display(centers_info[found_ids[found_idx], :])
        center = centers_info[found_ids[found_idx], 3:end]
        #display(center)
        # look for the correct hyperparameters:
        found_idx = nothing
        for i ∈ eachindex(ds_uids)
            if id_look == ds_uids[i]
                found_idx = i
                break
            end
        end
        featname, naf, nmf = ds[found_idx, [4, 5, 6]] # the data that we want to find
        featfile = ""
        if featname == "SOAP"
            featfile = "data/SOAP.jld"
        elseif featname == "ACSF"
            featfile = "data/ACSF.jld"
        end
        uid = id_look
        kid = id_look*"_"*best10[k, 2]
        #println([uid," ",kid])
        println([k, featname, naf, nmf, kid])
        # test one ds and fit
        data_setup("exp_reduced_energy", naf, nmf, 3, 300, "data/qm9_dataset_old.jld", featfile, featname)
        GC.gc()
        # loop the model:
        fit_🌹_and_atom("exp_reduced_energy", "data/qm9_dataset_old.jld";
                            model="GAK", E_atom=E_null, center_ids=center, uid=uid, kid=kid)
        GC.gc()
#=         for model ∈ models
            fit_🌹_and_atom("exp_reduced_energy", "data/qm9_dataset_old.jld";
                            model=model, E_atom=E_null, center_ids=center, uid=uid, kid=kid)
            GC.gc()
        end =#
    end
end


"""
big main function here, to tune hyperparameters by DFO
    
"""
function hyperparamopt(;init=false, init_data=[], init_x = [], dummyfx = false, trackx = true, fmode = "molecule")
    # initial fitting, initialize params and funs, replace with actual fitting:
    path_init_params = "data/hyperparamopt/init_params.txt"; 
    path_params = "data/hyperparamopt/params.txt"; 
    path_fun = "data/hyperparamopt/fun.txt"; 
    path_track="data/hyperparamopt/tracker.txt";
    path_bounds = "data/hyperparamopt/bounds.txt";
    bounds = readdlm(path_bounds)
    if dummyfx # if dummy function value is requested:
        fx = fxdummy
    else
        if fmode == "molecule"
            fx = main_obj
        elseif fmode == "atom"
            fx = main_obj_atom
        end
    end
    if init # init using arbitrary params (or init_data), e.g., the best one:
        uid = replace(string(Dates.now()), ":" => ".")
        if isempty(init_data)
            println("init starts, computing fobj...")
            #x = [20/51, 16/20, 3/10, 1/3, 1/1, 1/1, 38/95, 5/5, 1/32] # best hyperparam from pre-hyperparamopt exps
            #x = repeat([0.5], 9) # midpoint if continuosu
            #x = [0.5, 0.5, 6.0, 2.0, 0.0, 0.0, 48.0, 3.0, 524288.0] # midpoint
            if !isempty(init_x)
                x = init_x
            else # use predefined x
                x = [0.5, 0.5, 6.0, 2.0, 0.0, 0.0, 48.0, 4.0, 524288.0] # midpoint, shifted the model by 1
            end
            f = fx(x)
        else
            println("init starts, initial fobj and points known")
            x = init_data[2:end] # unencoded x, the one that Julia accepts
            f = init_data[1]
        end
        x = encode_parameters(x, bounds) # encode x for mintry
        data = Matrix{Any}(undef, 1, length(x)+1)
        data[1,1] = uid; data[1,2:end] = x 
        writestringline(string.(vcat(uid, x)), path_init_params); writestringline(string.(vcat(uid, x)), path_params)
        writestringline(string.(vcat(uid, f)), path_fun)
        println("init done")
    else
        # continue using prev params:
        data = readdlm(path_params)
        println("start hyperparamopt using previous checkpoint")
        # do fitting:
        x = data[1,2:end]
        f = fx(x)
        # write result to file:
        uid = replace(string(Dates.now()), ":" => ".")
        writestringline(string.(vcat(uid, f)), path_fun); 
        if trackx
            writestringline(string.(vcat(f, x)), path_track; mode="a");
        end
    end
    # tracker for fobj and hyperparams already used:
    #writestringline(string.(vcat(f, x)), path_track; mode="a") 
    while true
        prev_data = data; prev_f = f; # record previous data
        newdata = readdlm(path_params)
        if data[1,1] != newdata[1,1] # check if the uid is different
            uid = replace(string(Dates.now()), ":" => ".")
            if data[1,2:end] == newdata[1,2:end] # if the content is the same (it's posssible due to the random rounding algorithm), then return previous f
                writestringline(string.(vcat(uid, prev_f)), path_fun)
            else # compute new f, or if the point was already "seen", return the function value
                data = newdata
                println("new incoming data ", data)
                # do fitting:
                x = data[1,2:end]
                # check if x is already seen, if true, return (f,x):
                idx = nothing
                if filesize(path_track) > 0 && isfile(path_track)
                    tracker = readdlm(path_track)
                    for i in axes(tracker, 1)
                        if x == tracker[i, 2:end]
                            idx = i
                            break
                        end
                    end
                end
                if idx === nothing # if not found, compute new point and save to tracker
                    println("computing fobj from new point...")
                    f = fx(x)
                    if trackx
                        writestringline(string.(vcat(f, x)), path_track; mode="a")
                    end
                else
                    println("the point was already tracked!")
                    f = tracker[idx, 1]
                end
                # write result to file:
                writestringline(string.(vcat(uid, f)), path_fun)
                println("fobj computation finished")
                GC.gc(); GC.gc(); GC.gc(); # GC is working only in function context
            end
        end
        sleep(0.3)
    end
end

"""
simulators spawner for parallel hyperparam
= states: 0 = idle, 1 = running, 2 = killed (or killed is no file found)
- sim_id must be positive integers as low as possible due to the nature of the cell
- fobj_mode:    1 = no baseline fitting (only dressed atoms), 
                2 = with baseline fitting and preloads most variables, which should be faster, if this pass the unit test, this will be set as default
"""
function hyperparamopt_parallel(sim_id; dummyfx = false, trackx = true, fobj_mode = 1)
    println("simulator", sim_id, " has been initialized!!")
    # paths to necessary folders:
    path_f = "data/hyperparamopt/sim/f/"
    path_x = "data/hyperparamopt/sim/x/"
    path_tracker = "data/hyperparamopt/sim/sim_tracker.txt"
    if !isfile(path_tracker)
        writedlm(path_tracker, []) # if tracker doesn't exist, create empty
    end
    # preload dataset, E, DFs, Fs, centers; and compute idtrains, idtests if fobj_mode == 2
    if fobj_mode == 2
        dataset = load("data/qm9_dataset.jld", "data") # dataset info
        E = vec(readdlm("data/energies.txt")) # base energy
        Fa = load("data/atomref_features.jld", "data") # DA feature
        Fb = load("data/featuresmat_bonds_qm9_post.jld", "data") # DB
        Fn = load("data/featuresmat_angles_qm9_post.jld", "data") # DN
        Ft = load("data/featuresmat_torsion_qm9_post.jld", "data") # DT
        DFs = [Fa, Fb, Fn, Ft]
        feat_paths = ["data/ACSF_51.jld", "data/SOAP.jld", "data/FCHL19.jld", "data/MBDF.jld", "data/CMBDF.jld"]
        Fs = map(feat_paths) do fpath # high-level energy features
            load(fpath, "data")
        end
        # centers, idtrains, idtests:
        #= rank = 2 #select set w/ 2nd ranked training MAE
        id = Int(readdlm("result/deltaML/sorted_set_ids.txt")[rank])
        centers = Int.(readdlm("data/all_centers_deltaML.txt")[:, id]) =#

        minid = 57 # see "MAE_custom_CMBDF_centers", this gives 3.7kcal/mol for x = [0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 5, 0, 0, 5, 11, 2]
        centerss = readdlm("data/custom_CMBDF_centers_181023.txt", Int)
        centers = centerss[minid, :]

        idall = 1:length(E)
        idtrains = centers[1:100]
        idtests = setdiff(idall, idtrains)
    end

    # initialize simulator:
    if dummyfx
        fx = fxdummy
    else
        fx = main_obj
    end
    path_sim_f = path_f*"sim_$sim_id.txt"
    writedlm(path_sim_f, [0]) # init, state = "idle"

    # listen to path_x
    xinfo = xuid = nothing # initialize data structures
    path_sim_x = path_x*"sim_$sim_id.txt"
    try
        while true
            if filesize(path_sim_x) > 1 && isfile(path_sim_x) # check if file is not empty
                try # for some reason sometimes the written x is empty, HELLO OCTAVE??
                    xinfo = readdlm(path_sim_x)
                catch ArgumentError
                    println("ERROR: x info is empty!")
                    continue
                end
                if xuid != xinfo[1] # check if the uid is different from the previous one
                    writedlm(path_sim_f, [1]) # state = "running"
                    xuid = xinfo[1]; iter = xinfo[2]; x = xinfo[3:end] # get x info
                    println("new incoming xinfo!", x)
                    # find if x is in the repo/tracker:
                    idx = nothing
                    if filesize(path_tracker) > 1
                        tracker = readdlm(path_tracker)
                        for i ∈ axes(tracker, 1)
                            if x == tracker[i, 4:end] # [sim_id, fuid, f, x]
                                idx = i
                                break
                            end
                        end
                    end
                    fuid = rand(1)[1] # random fuid, IS A VECTOR!, hence take the first elem only
                    if idx !== nothing # if x is found in the repo, then just return the f given by the index
                        println("x found in tracker!")
                        f = tracker[idx, 3]
                    else
                        println("x not found in tracker, computing f(x)...")
                        # compute f=f(x):
                        if fobj_mode == 1
                            f = fx(x; sim_id = "_$sim_id")
                        elseif fobj_mode == 2
                            f = fx(E, dataset, DFs, Fs, centers, idtrains, x; sim_id = "_$sim_id")
                        end
                        # write to tracker:
                        if trackx 
                            writestringline(string.(vcat(sim_id, fuid, f, x)'), path_tracker; mode="a") # [fuid, f, x]
                        end
                    end
                    # write f info to controller listener:
                    println("new f info has been written")
                    writedlm(path_sim_f, [0, iter, fuid, f]', "\t") # [state, iter, fuid, f] 
                    println("waiting for new x...")
                end
            end
            sleep(.05) # delay a  bit for harddisk
        end 
    catch exc # add error catcher if simulator is killed or something, then remove the f info file (signal sender)
        if exc isa InterruptException
            println("simulator is killed!")
        elseif exc isa OutOfMemoryError
            println("OOM!")
        else
            throw(exc)
        end
        rm(path_sim_f)
    end
end

"""
encode parameters to desired form by the solver
particularly categorical variables are encoded by one-hot-encoding
"""
function encode_parameters(x, bounds)
    xout = []
    for i ∈ eachindex(x) # for each variable:
        if bounds[3, i] == 2 # check for type, handle categorical (2)
            # one hot encoding:
            len = Int(bounds[2, i] - bounds[1, i] + 1)
            values = LinRange(Int(bounds[1, i]), Int(bounds[2, i]), len)
            v = zeros(Int(len))
            v[findfirst(c -> c == x[i], values)] = 1
            append!(xout, v)
        else
            push!(xout, x[i])
        end
    end
    return xout
end

"""
processes the raw parameters from mintry, then compute the MAE from the given parameters
par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
par_fit_atom = [center_ids] # center_ids = 0 → use 
par_fit = [model, cσ]
params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
           1       2   3           4           5               6               7       8       9
bounds = [1,var], [1,var], [1,10], [1,3],   [0,1],          [0,1]           [0,95],    [1,5], [1,1e+10]     
naf, nmf in percentage, e.g., .5 -> .5*max(naf("ACSF"))
feature name: 1=ACSF, 2=SOAP, 3=FCHL
model: ["ROSEMI", "KRR", "NN", "LLS", "GAK", "REAPER"]

new params:
[n_mf, n_af, n_basis, feature_name, normalize_atom, normalize_mol, model, c]
    1   2       3       4               5               6           7     8 
the rest of the definitions are the same 
"""
function main_obj(x; sim_id="")
    # process params:

    # determine n_af and n_mf:
    n_mf = Int(x[1]); n_af = Int(x[2]);
    n_basis = Int(x[3]) # determine number of splines

    # determine feature_name and path:
    dftype = Dict()
    dftype[1] = "ACSF_51"; dftype[2] = "SOAP"; dftype[3] = "FCHL19";
    feature_name = dftype[Int(x[4])]; feature_path = "data/"*feature_name*".jld";

    # get centers indices:
    rank = 2 #select set w/ 2nd ranked training MAE
    id = Int(readdlm("result/deltaML/sorted_set_ids.txt")[rank])
    centers = Int.(readdlm("data/all_centers_deltaML.txt")[:, id])
    E = vec(readdlm("data/E_clean_sorted.txt")[:, 2]) # load Etarget

    # determine normalize switches:
    norms = Dict()
    norms[0] = false; norms[1] = true
    normalize_atom = norms[Int(x[5])]; normalize_mol = norms[Int(x[6])];
    # determine model:
    lmodel = ["ROSEMI", "KRR", "NN", "LLS", "GAK", "REAPER"]
    model = lmodel[Int(x[7])]
    c = 2^x[8] # determine gaussian scaler for kernels

    println([n_mf, n_af, n_basis, feature_name, normalize_atom, normalize_mol, feature_path, model, c])

    foldername = "exp_hyperparamopt_"*sim_id; file_dataset = "data/qm9_dataset.jld";
    
    # compute feature transformaiton and data selection, the centerss output ended up not being used for current version, due to the centers are already predetermined
    F, f, centerss, ϕ, dϕ, dataset = data_setup(foldername, n_af, n_mf, n_basis, 300, file_dataset, feature_path, feature_name; 
                                normalize_atom = normalize_atom, normalize_mol = normalize_mol, save_global_centers = false, num_center_sets = 1, save_to_disk = false)
    full_fit_🌹(E, dataset, F, f, centers, ϕ, dϕ, foldername; 
                bsize = 1000, tlimit = 900, model = model, ca = c, cm = c)
    # get MAE:
    path_result = "result/$foldername/err_$foldername.txt"
    MAE = readdlm(path_result)[end, 5] # take the latest one on the 5th column

    F = f = centerss = ϕ = dϕ = nothing # clear var
    GC.gc() # always gc after each run
    return MAE
end

"""
main objective augmented with Elevel
mandatory params:
    - E, energy vector
    - FA, FB, FN, FT (all dressed features), vector of all dressed features
    - f_atom, all sets of atomic feautres (possibly) over 20gbs
    - dataset
    - centers
hyperparams (for optimization, under one vector x):
    1. Edb_switch ∈ cat[0,1], whether to include Edb or not
    2. Edn ...
    3. Edt ...
    4. Edb_PCA_switch ∈ cat[0,1], whethre to do PCA or not for the Edb
    5. Edn_PCA_switch ∈ cat[0,1] ...
    6. Edt_PCA_switch ∈ cat[0,1] ...
    7. num_Edb_f, number of E_db features outputed by PCA ∈ int[1,10]
    8. num_Edn_f, ...
    9. num_Edt_f, ...
    10. fmol_PCA_switch ∈ -> cat[0,1]
    11. fatom_PCA_switch, ...
    12. num_fmol ∈ oint[1,50]
    13. num_fatom ...
    14. n_basis ∈ int[1,10]
    15. feature_name ∈ int[1,5] -> cat
    16. normalize_atom ∈ cat[0,1]
    17. normalize_mol ...
    18. model ∈ int[1,6] -> cat
    19. const ∈ int[1,20]
    20. solver ∈ int[1,2] -> cat
"""
function main_obj(E, dataset, DFs, Fs, centers, idtrains, x; sim_id = "")
    x = Int.(x) # cast to integers
    # PCA and fit DFs:
    bools = [false, true]
    println(x)
    Et = hp_baseline(E, DFs..., idtrains; 
                sb=bools[x[1]+1], sn=bools[x[2]+1], st=bools[x[3]+1],
                pb=bools[x[4]+1], pn=bools[x[5]+1], pt=bools[x[6]+1],
                npb = x[7], npn = x[8], npt = x[9])
    # data setup:
    pca_atom = bools[x[10]+1]; pca_mol = bools[x[11]+1]
    # determine n_af and n_mf:
    n_mf = Int(x[12]); n_af = Int(x[13]);
    # for now, do a heavisidestep if feature = MBDF since |MBDF| = 6; |CMBDF| = 40:
    if x[15] == 4
        n_mf = min(n_mf, 6); n_af = min(n_af, 6)
    elseif x[15] == 5 || x[15] == 6
        n_mf = min(n_mf, 40); n_af = min(n_af, 40)
    end
    n_basis = Int(x[14]) # determine number of splines
    # determine feature:
    ftypes = ["ACSF_51", "SOAP", "FCHL19", "MBDF", "CMBDF", "CMBDF2"]
    feature = Fs[x[15]]; feature_name = ftypes[x[15]]
    # switches:
    normalize_atom = bools[Int(x[16]) + 1]
    normalize_mol = bools[Int(x[17]) + 1]
    # model params:
    lmodel = ["ROSEMI", "KRR", "NN", "LLS", "GAK", "REAPER"]
    model = lmodel[Int(x[18])]
    c = 2. ^x[19]
    # solver params:
    solvers = ["cgls", "direct"]
    solver = solvers[x[20]]
    println([bools[x[1]+1], bools[x[2]+1], bools[x[3]+1], bools[x[4]+1], bools[x[5]+1], bools[x[6]+1], 
            x[7], x[8], x[9], pca_atom, pca_mol, n_mf, n_af, n_basis, feature_name,
            normalize_atom, normalize_mol, model, c, solver])
    # compute feature transformaiton and data selection, the centerss output ended up not being used for current version, due to the centers are already predetermined
    foldername = "exp_hyperparamopt_"*sim_id;
    F, f, centerss_out, ϕ, dϕ = data_setup(foldername, n_af, n_mf, n_basis, 10, dataset, feature, feature_name; 
                                        pca_atom = pca_atom, pca_mol = pca_mol, normalize_atom = normalize_atom, normalize_mol = normalize_mol, 
                                        save_global_centers = false, num_center_sets = 1, save_to_disk = false)
    
    # fit fatoms: 
    full_fit_🌹(Et, dataset, F, f, centers, ϕ, dϕ, foldername; 
                bsize = 1000, tlimit = 900, model = model, solver = solver, ca = c, cm = c)
    # get MAE:
    path_result = "result/$foldername/err_$foldername.txt"
    MAE = readdlm(path_result)[end, 5] # take the latest one on the 5th column

    F = f = centerss_out = ϕ = dϕ = nothing # clear var
    GC.gc() # always gc after each run
    return MAE
end

"""
ATOM ONLY!
params = [naf, fname, norm_atom, model, c]
"""
function main_obj_atom(x)
    # process params:

    # determine n_af and n_mf:
    n_af = Int(x[1]);

    # determine feature_name and path:
    dftype = Dict()
    dftype[1] = "ACSF"; dftype[2] = "SOAP"; dftype[3] = "FCHL";
    feature_name = dftype[Int(x[2])]; feature_path = "data/"*feature_name*".jld";

    # crawl center_id by index, "data/centers.txt" must NOT be empty:
    centers = readdlm("data/centers.txt")
    center_idx = 38 # set it fixed as current "best" found training points
    uid=""; kid= ""; uk_id = ""
    uid = centers[center_idx,1]; kid = centers[center_idx,2]; uk_id = join([uid,"_",kid])
    center = centers[center_idx, 3:end]
    # get atom info global, to match center ids:
    atomref = readdlm("data/atomref_info.txt")
    f_atom = load("data/atomref_features.jld", "data")
    E_atom = f_atom*atomref[center_idx,5:end]

    # determine normalize switches:
    norms = Dict()
    norms[0] = false; norms[1] = true
    normalize_atom = norms[Int(x[3])];
    # determine model:
    lmodel = ["ROSEMI", "KRR", "NN", "LLS", "GAK", "REAPER"]
    model = lmodel[Int(x[4])]
    c = 2^x[5] # determine gaussian scaler for kernels

    println([n_af, feature_name, normalize_atom, normalize_mol, feature_path, model, c])

    foldername = "exp_hyperparamopt"; file_dataset = "data/qm9_dataset_old.jld"; file_atomref_features = "data/atomref_features.jld"
    
    data_setup(foldername, n_af, n_mf, n_basis, 300, file_dataset, feature_path, feature_name; 
        normalize_atom = normalize_atom, normalize_mol = normalize_mol, save_global_centers = true, num_center_sets = 1)
    GC.gc() # always gc after each run
    fit_🌹_and_atom(foldername, file_dataset; model = model, 
        E_atom = E_atom, cσ = c, scaler = c, 
        center_ids = center, uid = uid, kid = uk_id)
    # get MAE:
    path_result = "result/$foldername/err_$foldername.txt"
    MAE = readdlm(path_result)[end, 5] # take the latest one on the 5th column

    f_atom = E_atom = nothing # clear var
    GC.gc() # always gc after each run
    return MAE
end

function fxdummy(x; sim_id="")
    u = 0.
    s = norm(x .- u)^2
    return 10. + (5s)/(s+1)
end

"""
project parameters to boudaries, given 2 × n boundary matrix
"""
function parambound!(x, bounds)
    # round x by probablity:
    for i ∈ eachindex(x)[3:end]
        p = rand(1);
        xl = floor(x[i]);
        f = x[i] - xl;
        if p < 1-f; # larger chance to be rounded to the closest int
            x[i] = xl;
        else
            x[i] = ceil(x[i]);
        end
    end
    # clip to boundary:
    for i ∈ eachindex(x)
        x[i] = max(bounds[1,i], min(x[i], bounds[2,i]))
    end
end

"""
julia version of hyperparameteropt
"""
function hyperparamopt_jl()
    path_tracker="data/hyperparamopt/tracker_jl.txt"; path_best = "data/hyperparamopt/best_jl.txt"
    f(naf, nmf, ns, fn, na, nm, cid, model, c) = main_obj([naf, nmf, ns, fn, na, nm, cid, model, c]) # f(x) wrapper
    # only RandomSampler() will automatically get the best point if canceled (ctrl+c)
    # initialize trackers:
    if isfile(path_tracker)
        println("previous tracker exists!")
        indata = readdlm(path_tracker)
        fs = indata[:, 1]; xs = indata[:, 2:end]; xs = [xs[i,:] for i ∈ axes(xs, 1)]
    else
        println("tracker does not exist, initializing!")
        fs = []; xs = []
    end
    # hyperparameteropt:
    # declare the possible values, try fixing ns=3, model=5, na=1, nm=1, cid=38:
    ns=3/5; model=5/5; na=1/1; nm=1/1; cid=38/95;
    ho = @hyperopt for resources=100, sampler=Hyperband(R=100, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()])), 
    naf = LinRange(0,1,101), nmf = LinRange(0,1,101),
    fn = [1/3, 2/3, 3/3], c = LinRange(0,1,101)
        #sleep(1)
        print(resources,"\t")
        if !(state === nothing)
            naf, nmf, fn, c = state
        end
        xv = [naf, nmf, fn, c]
        xid = findfirst(xel->xel == xv, xs)
        if xid !== nothing # if prev saved iterate is found, return the found point
            println("prev point found!")
            fv = fs[xid]
        else # new point, compute new objective
            @show fv = f(naf, nmf, ns, fn, na, nm, cid, model, c)
            push!(fs, fv); push!(xs, xv) # tracker
            open(path_tracker, "a") do io writedlm(io, transpose(vcat(fv, xv))) end # write to file directly too, in case of execution kill
        end
        fv, xv # return values (f,x)
    end
    #best_params, min_f = ho.minimizer, ho.minimum # this returns something only if it's not abruptly stopped
    # check best:
    minf, minid = findmin(fs)
    minimizer = xs[minid]
    display([minf, minimizer])
    if isfile(path_best)
        best = readdlm(path_best)
        if minf < best[1,1] # write the new best
            writedlm(path_best, transpose(vcat(minf, minimizer)))
        end
    else # write immediately, since it's empty
        writedlm(path_best, transpose(vcat(minf, minimizer)))
    end
    # write tracker:
    xs = mapreduce(permutedims, vcat, xs)
    out = hcat(fs, xs)
    writedlm(path_tracker, out)
end


"""
to fit ex-datasets, i.e., dataset with excluded uncharacterized molecules.
"""
function fit_ex()
    foldername = "exp_ex"
    # ACSF:
    nafs = [40, 30, 20] 
    nmfs = [40, 30, 20]
    println("ACSF")
    for i ∈ eachindex(nafs)
        for j ∈ eachindex(nmfs)
            nmf = nmfs[j]
            if nmf >= nafs[i] # the num of mol feature must be less than atom f
                nmf = nafs[i] - rand(1:5, 1)[1]
            end
            println(nafs[i]," ",nmf)
            data_setup(foldername, nafs[i], nmf, 3, 300, "data/qm9_dataset_old.jld", "data/ACSF.jld", "ACSF"; 
                        save_global_centers = true, num_center_sets = 5)
            GC.gc()
        end
    end
end

function test_BOHB()
    path_tracker="data/hyperparamopt/tracker_jl.txt"; path_best = "data/hyperparamopt/best_jl.txt"
    f(aa, x,c) = sum(x^2 + c^2) 
    bohb = @hyperopt for i=18, sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])), x = LinRange(1,5,800), c = exp10.(LinRange(-1,3,1800))
        sleep(1)
        if state !== nothing
            x,c = state
        end
        print(i,"\t")
        @show fv = f(0, x,c)
        fv, [x,c]
    end
end

"""
sanity check whenever there is a change in the Julia env (in particular in VSC)
"""
function sanity_check()
    println("SANITY CHECK!")
    centers = readdlm("data/centers.txt")[38, 3:end]
    E_atom = readdlm("data/atomic_energies.txt")
    data_setup("exp_reduced_energy", 20, 16, 3, 100, "data/qm9_dataset_old.jld", "data/ACSF.jld", "ACSF"; save_global_centers = false, num_center_sets=1)
    fit_atom("exp_reduced_energy", "data/qm9_dataset_old.jld", "data/atomref_features.jld"; center_ids=centers)
    fit_🌹_and_atom("exp_reduced_energy", "data/qm9_dataset_old.jld"; model = "REAPER", E_atom = E_atom, cσ = 2048., scaler = 2048., center_ids = centers)
    println("SANITY CHECK DONE!")
end



function test_mainobj()
    # simulate input parameters:
    sim_id = "sanitytest"
    #x = [1, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 1, 0, 0, 6, 11] # current best conf found w.r.t the current hyperparameter space, 7.59 kcal/mol
    #x = [0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 4, 0, 0, 5, 11, 2] # current best conf found w.r.t the current hyperparameter space, 5.78 kcal/mol
    x = [0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 5, 0, 0, 5, 11, 2] # current best conf found w.r.t the current hyperparameter space, 5.03 kcal/mol
    #x = [0, 1, 0, 0, 0, 1, 10, 10, 10, 0, 0, 50, 50, 1, 5, 0, 0, 5, 4, 2] # 4.5kcal/mol
    # inside functions:
    dataset = load("data/qm9_dataset.jld", "data") # dataset info
    E = vec(readdlm("data/energies.txt")) # base energy
    Fa = load("data/atomref_features.jld", "data") # DA feature
    Fb = load("data/featuresmat_bonds_qm9_post.jld", "data") # DB
    Fn = load("data/featuresmat_angles_qm9_post.jld", "data") # DN
    Ft = load("data/featuresmat_torsion_qm9_post.jld", "data") # DT
    DFs = [Fa, Fb, Fn, Ft]
    feat_paths = ["data/ACSF_51.jld", "data/SOAP.jld", "data/FCHL19.jld", "data/MBDF.jld", "data/CMBDF.jld"]
    Fs = map(feat_paths) do fpath # high-level energy features
        load(fpath, "data")
    end
    #Fs = [[],[],[],load(feat_paths[4], "data")]
    # centers, idtrains, idtests:
    #= rank = 2 #select set w/ 2nd ranked training MAE
    id = Int(readdlm("result/deltaML/sorted_set_ids.txt")[rank])
    centers = Int.(readdlm("data/all_centers_deltaML.txt")[:, id]) =#
    # custom CMBDF training sets:
    minid = 57 # see "MAE_custom_CMBDF_centers"
    centerss = readdlm("data/custom_CMBDF_centers_181023.txt", Int)
    centers = centerss[minid, :]
    idtrains = centers[1:100]
    fx = main_obj
    f = fx(E, dataset, DFs, Fs, centers, idtrains, x; sim_id = "_$sim_id")
    writedlm("test_mainobj.txt", f)
end

"""
since we figured out that CMBDF gives the best MAE ~4.5 kcal/mol with selection where CMBDF WAS NOT INCLUDED, so now we want to see if the data seleciton uses MBDF what would happen
"""
function main_custom_CMBDF_train()
    f = load("data/CMBDF2.jld", "data")
    nrow = length(f)
    # try using flatten instead of the usual sum, feels like summing causes some information lost:
    ncol = 29*40
    F = zeros(Float64, nrow, ncol)
    for i ∈ 1:nrow
        F[i,eachindex(f[i])] = vec(transpose(f[i])) # flatten
    end
    Random.seed!(777)
    nset = 1_000
    # with selection algo:
    centerss = set_cluster(F, 200; universe_size = 1000, num_center_sets = nset)
    writedlm("data/custom_CMBDF2_centers_191023.txt", centerss)
    # random:
    #= centerss = [sample(1:nrow, 200, replace=false) for i ∈ 1:nset]
    writedlm("data/random-777_CMBDF2_centers_191023.txt", centerss) =#

    centerss = Int.(transpose(reduce(hcat, centerss))) # transform to matrix

    # run it all through the base "main_obj":
    # spawn all memory dependednt data:
    sim_id = "custom_CMBDF2_centers_191023"
    #x = [0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 5, 0, 0, 5, 11, 2] # current best conf found w.r.t the current hyperparameter space, 5.03 kcal/mol
    x = [0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 50, 50, 3, 6, 0, 0, 5, 11, 2] # try CMBDF2
    # inside functions:
    dataset = load("data/qm9_dataset.jld", "data") # dataset info
    E = vec(readdlm("data/energies.txt")) # base energy
    Fa = load("data/atomref_features.jld", "data") # DA feature
    Fb = load("data/featuresmat_bonds_qm9_post.jld", "data") # DB
    Fn = load("data/featuresmat_angles_qm9_post.jld", "data") # DN
    Ft = load("data/featuresmat_torsion_qm9_post.jld", "data") # DT
    DFs = [Fa, Fb, Fn, Ft]
    feat_paths = ["data/ACSF_51.jld", "data/SOAP.jld", "data/FCHL19.jld", "data/MBDF.jld", "data/CMBDF.jld", "data/CMBDF2.jld"]
    Fs = [[],[],[],[],[],load(feat_paths[5], "data")]
    fx = main_obj
    for i ∈ axes(centerss, 1) # centers, idtrains, idtests:
        centers = centerss[i,:]
        idtrains = centers[1:100]
        fobj = fx(E, dataset, DFs, Fs, centers, idtrains, x; sim_id = "_$sim_id")
        strinput = string.([fobj])
        writestringline(strinput, "result/deltaML/MAE_custom_CMBDF2_centers_191023.txt"; mode="a")
        #writestringline(strinput, "result/deltaML/MAE_random-777_CMBDF_centers_181023.txt"; mode="a")
    end
end


"""
julia GC test, aparently only work within function context, not outside
"""
function gctest()
    while true
        println("init A")                                                                                                    
        A = rand(Int(sx), Int(sy))                                                                                       
        sleep(4)                                                                                                      
        A = nothing
        GC.gc()
        println("cleared A")
        sleep(4)                                                                                        
    end
end


"""
test race condition of multiple julia accessing one file:
"""
function test_race(id)
    iter = 0
    while true
        data = [id, string(iter)]
        println(data, " has been written")
        writestringline(data, "testrace.txt", mode="a")
        iter+=1
        sleep(.5)
    end
end





"""
dummy obj function given kernel K(H) with fixed hyperparameters H
"""
function fobj(x; A=zeros(), b=zeros(), grange=[])
    #= trainids = grange[x]
    testids = setdiff(grange, trainids)
    θ = A[trainids,:]\b[trainids] # train on index x
    bpred = A[testids,:]*θ
    return norm(bpred-b) =# # error of the test 
end

"""
simple test for JuMP NLP with constraints
"""
function test_JUMP()
    # ================
    # case 1: simple LP but solved by NLP (Ipopt):
    # dummy fobj:
    #= function fobj(x,y;h=0.)
        return 12*x + 20*y + h
    end
    model = Model()
    register(model, :fobj, 2, fobj; autodiff=true) # register custom fobj, number of optimized variables = 2
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @constraint(model, c1, 6x + 8y >= 100)
    @constraint(model, c2, 7x + 12y >= 120)
    @objective(model, Min, fobj(x,y;h=1000))
    set_optimizer(model, Ipopt.Optimizer)
    optimize!(model)
    display(objective_value(model))
    display([value(x), value(y)]) =#

    # ===============
    # case 2: dummy underdetermined lsq, binary selection of array
    Random.seed!(777)
    A = rand(100, 20)
    b = rand(100)
    grange = 1:length(b)
    function fobj(x::T...; A=zeros(), b=zeros(), grange=[]) where {T<:Real}
        #= trainids = grange[x]
        testids = setdiff(grange, trainids)
        θ = A[trainids,:]\b[trainids] # train on index x
        bpred = A[testids,:]*θ
        return norm(bpred-b) =# # error of the test 

        Dx = diagm(x)
        θ = (Dx*A)\(Dx*b)
        nx = similar(x)
        for i ∈ eachindex(x)
            if x[i] == 0
                nx[i] = 1
            elseif x[i] == 1
                nx[i] = 0
            end
        end
        Dnx = diagm(nx)
        return norm(Dnx*A*θ - Dnx*b)
    end
    model = Model()
    register(model, :fobj, 1, fobj; autodiff=true) # register custom fobj, number of optimized variables = 2
    @variable(model, x[1:length(b)], Bin)
    @constraint(model, sum(x) == 10)
    @objective(model, Min, fobj(x;A=A, b=b, grange=grange))
    set_optimizer(model, Ipopt.Optimizer)
    optimize!(model)
    display(objective_value(model))
    display([value(x)])
end

function test_NOMAD()
    # blackbox
    function bb(x; c=0.)
        f = (x[1]- 1)^2 * (x[2] - x[3])^2 + (x[4] - x[5])^2 +c
        bb_outputs = [f]
        success = true
        count_eval = true
        return (success, count_eval, bb_outputs)
    end
    # linear equality constraints
    #= A = [1.0 1.0 1.0 1.0 1.0;
        0.0 0.0 1.0 -2.0 -2.0]
    b = [5.0; -3.0]
    =#
    A = [1.0 1.0 1.0 1.0 1.0]
    b = [5.0]

    # Define blackbox
    p = NomadProblem(5, 1, ["OBJ"], # the equality constraints are not counted in the outputs of the blackbox
                    x->bb(x;c=10.);
                    input_types = repeat(["B"], 5),
                    #lower_bound = -10.0 * ones(5),
                    #upper_bound = 10.0 * ones(5),
                    A = A, b = b)

    # Fix some options
    p.options.max_bb_eval = 500

    # Define starting points. It must satisfy A x = b.
    x0 = [0.57186958424864897665429452899843;
        4.9971472653643420613889247761108;
        -1.3793445664086618762667058035731;
        1.0403394252630473459930726676248;
        -0.2300117084673765077695861691609]
    
    #x0 = [1;0;0;1;0]

    # Solution
    result = solve(p, x0)
    println("Solution: ", result.x_best_feas)
    println("Satisfy Ax = b: ", A * result.x_best_feas ≈ b)
    println("And inside bound constraints: ", all(-10.0 .<= result.x_best_feas .<= 10.0))
end