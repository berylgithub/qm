using Hyperopt

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
function hyperparamopt(;init=false, init_data=[])
    # test "threading" loop:
    # initial fitting, initialize params and funs, replace with actual fitting:
    path_params = "data/hyperparamopt/params.txt"; path_fun = "data/hyperparamopt/fun.txt"; path_track="data/hyperparamopt/tracker.txt";
    if init # init using arbitrary params (or init_data), e.g., the best one:
        uid = replace(string(Dates.now()), ":" => ".")
        if isempty(init_data)
            println("init starts, computing fobj...")
            x = [20/51, 16/20, 3, 1, 1, 1, 38, 5, 32.0] # best hyperparam from pre-hyperparamopt exps
            f = main_obj(x)
        else
            println("init starts, initial fobj and points known")
            x = init_data[2:end]
            f = init_data[1]
        end
        data = Matrix{Any}(undef, 1, length(x)+1)
        data[1,1] = uid; data[1,2:end] = x 
        writestringline(string.(vcat(uid, x)), path_params)
        writestringline(string.(vcat(uid, f)), path_fun)
        println("init done")
    else
        # continue using prev params:
        data = readdlm(path_params)
        println("start hyperparamopt using previous checkpoint")
        # do fitting:
        x = data[1,2:end]
        f = main_obj(x)
        # write result to file:
        uid = replace(string(Dates.now()), ":" => ".")
        writestringline(string.(vcat(uid, f)), path_fun)
    end
    # tracker for fobj and hyperparams already used:
    writestringline(string.(vcat(f, x)), path_track; mode="a") 
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
                tracker = readdlm(path_track)
                idx = nothing
                for i in axes(tracker, 1)
                    if x == tracker[i, 2:end]
                        idx = i
                        break
                    end
                end
                if idx === nothing # if not found, compute new point and save to tracker
                    println("computing fobj from new point...")
                    f = main_obj(x)
                    writestringline(string.(vcat(f, x)), path_track; mode="a")
                else
                    println("the point was already tracked!")
                    f = tracker[i, 1]
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
processes the raw parameters from mintry, then compute the MAE from the given parameters
par_ds = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol] # this will be supplied by dfo driver
par_fit_atom = [center_ids] # center_ids = 0 → use 
par_fit = [model, cσ]
params = [n_af, n_mf, n_basis, feature_name, normalize_atom, normalize_mol,center_ids,model, cσ]
           1       2   3           4           5               6               7       8       9
naf, nmf in percentage, e.g., .5 -> .5*max(naf("ACSF"))
feature name: 1=ACSF, 2=SOAP, 3=FCHL
model: ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
"""
function main_obj(x)
    #display(x)
    # process params:
    # determine n_basis:
    max_ns = 10; ns = Int(max(1., round(x[3]*max_ns)))
    # determine feature_name and path:
    dftype = Dict()
    dftype[1] = "ACSF"; dftype[2] = "SOAP"; dftype[3] = "FCHL";
    fname_num = Int(max(1., round(x[4]*3.))) ;feature_name = dftype[fname_num]; feature_path = "data/"*feature_name*".jld";
    # determine n_af and n_mf:
    dfnaf = Dict() # determines max n_af
    dfnaf[1] = 51 # ACSF
    dfnaf[2] = 165 # SOAP
    dfnaf[3] = 140 # FCHL
    max_naf = dfnaf[fname_num]; n_af = Int(max(1, round(max_naf*x[1]))); n_mf = Int(max(1, round(n_af*x[2])))
    # crawl center_id by index, "data/centers.txt" must NOT be empty:
    centers = readdlm("data/centers.txt")
    maxlen = 95 # this is the total number of precomputed instance #size(centers, 1)
    center_idx = Int(round(x[7]*maxlen))
    uid=""; kid= ""; uk_id = ""
    if 0 < center_idx ≤ maxlen 
        uid = centers[center_idx,1]; kid = centers[center_idx,2]; uk_id = join([uid,"_",kid])
        center = centers[center_idx, 3:end]
        # get atom info global, to match center ids:
        atomref = readdlm("data/atomref_info.txt")
        f_atom = load("data/atomref_features.jld", "data")
        E_atom = f_atom*atomref[center_idx,5:end]
        #display(atomref[center_idx,5:end])
        #display(E_atom)
    else
        center = [] # default empty, this will crawl the center given by the data setup
        E_atom = []
    end
    # determine normalize switches:
    norms = Dict()
    norms[0] = false; norms[1] = true
    normalize_atom = norms[Int(round(x[5]))]; normalize_mol = norms[Int(round(x[6]))];
    # determine model:
    lmodel = ["ROSEMI", "KRR", "NN", "LLS", "GAK"]
    model = lmodel[Int(max(1., round(x[8]*5.)))]
    cσ = float(min(1e+10,round(1. / x[9]))) # Gausssian scaler

    println([n_af, n_mf, ns, feature_name, normalize_atom, normalize_mol, center_idx, model, cσ])
    #= println([uid, kid, uk_id])
    display(center)
    display(cσ) =#

    #= foldername = "exp_hyperparamopt"; file_dataset = "data/qm9_dataset_old.jld"; file_atomref_features = "data/atomref_features.jld"
    data_setup(foldername, n_af, n_mf, ns, 300, file_dataset, feature_path, feature_name; 
        normalize_atom = normalize_atom, normalize_mol = normalize_mol, save_global_centers = true, num_center_sets = 2)
    GC.gc() # always gc after each run
    fit_atom(foldername, file_dataset, file_atomref_features; center_ids=center, uid=uid, kid=kid, save_global=true)
    GC.gc() # always gc after each run
    fit_🌹_and_atom(foldername, file_dataset; model = model, 
        E_atom = E_atom, cσ = cσ, scaler = cσ, 
        center_ids = center, uid = uid, kid = uk_id)
    # get MAE:
    path_result = "result/$foldername/err_$foldername.txt"
    MAE = readdlm(path_result)[end, 5] # take the latest one on the 5th column
    f_atom = E_atom = nothing # clear var
    GC.gc() # always gc after each run
    return MAE =#
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
    ns=3; model=5; na=1; nm=1; cid=38;
    ho = @hyperopt for resources=256_000, sampler=Hyperband(R=256_000, η=3, inner=RandomSampler()), 
    naf = LinRange(0,1,100), nmf = LinRange(0,1,100),
    fn = LinRange(1,3,3), c = LinRange(1,256,256)
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

# script to write string given a vector{string}
function writestringline(strinput, filename; mode="w")
    open(filename, mode) do io
        str = ""
        for s ∈ strinput
            str*=s*"\t"
        end
        print(io, str*"\n")
    end
end

# script to write to latex table, given a Matrix{Any}
function writelatextable(table, filename)
    open(filename, "w") do io
        for i ∈ axes(table, 1)
            str = ""
            for j ∈ axes(table, 2)
                str *= string(table[i, j])*"\t"*"& "
            end
            str = str[1:end-2]
            str *= raw"\\ \hline"*"\n"
            print(io, str)
        end
    end
end

"""
reference function to query any type of data, the queries are similar to this
"""
function dataqueryref()
    # query example:
    for i ∈ axes(best10, 1)
        for j ∈ axes(centers, 1)
            if best10[i, 1] == centers[j,1] && best10[i, 2] == centers[j, 2]
                push!(kidx, j)
                break
            end
        end
    end
    # string formatting example:
    table_null[2:end, end-1:end] = map(el -> @sprintf("%.3f", el), table_null[2:end, end-1:end])
    # table header example:
    table_exp[1,:] = [raw"\textbf{MAE}", raw"\textbf{\begin{tabular}[c]{@{}c@{}}Null \\ train \\MAE\end{tabular}}", raw"\textbf{model}", raw"$k$", raw"$f$", raw"$n_{af}$", raw"$n_{mf}$", raw"$t_s$", raw"$t_p$"]
    # query from setup_info.txt example:
    for i ∈ axes(table_k, 1)
        table_k[i, 2] = datainfo[didx[(i-1)*5 + 1], 4]
    end
    # taking mean example:
    for i ∈ axes(table_k, 1)
        table_k[i, 5] = mean(atominfo[(i-1)*5+1:(i-1)*5+5, 4])
    end
    # filling str with latex interp example:
    for i ∈ eachindex(cidx)
        table_centers[1, i] = L"$k=$"*string(cidx[i])
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
