using DelimitedFiles, Random, StatsBase

include("utils.jl")

"""
=========================
Tabu search w/ penalty
=========================
"""

"""
compute penalty given new f,u,opt
"""
function f_penalty(f, u, opt)
    (f + opt) / (u + 1)
end


"""
function to initialize fobj contrib and contrib count
    - x is the data index
    - f is the ∑fobj so far
    - u is the count so far
"""
function init_penalty_x(x, f, u, opt, tb_maes, tb_centers)
    for i ∈ eachindex(tb_maes) # for each row
        if x ∈ tb_centers[i,:]
            f += tb_maes[i] # update fobj
            u += 1 # update count
        end
    end
    return f_penalty(f, u, opt), f, u
end

"""
vectorized and binarized version, should be much faster
    - ids_int ∈ vector{Int} length = ndata
    - tb_centers converted to binaries, to avoid index search
"""
function init_penalties_x!(ps, fs, us, ids_int, opt, tb_maes, tb_centers)
    for x ∈ ids_int 
        for i ∈ eachindex(tb_maes)
            if tb_centers[i,x] # tb_centers is binaries
                fs[x] += tb_maes[i] # sum the fobj involved
                us[x] += 1  # increment count where x is involved
            end
        end
    end
    ps[ids_int] .= f_penalty.(fs[ids_int], us[ids_int], opt) # compute penalty
end

"""
update the table values (p,f,u) given new score fobj and training set S
    - fobj ∈ R, opt ∈ R
    - S ∈ vector[0,1] length = ndata (binary)
    - ps, fs, us, ∈ vectors{R} length = ndata
"""
function update_penalties_x!(ps, fs, us, fobj, opt, S)
    fs[S] .+= fobj # increase fobj value
    us[S] .+= 1 # increment counter
    ps[S] .= f_penalty.(fs[S], us[S], opt)
end

"""
updates the set S: replaces some x∈S (up to m numbers) that has high penalty with some x∉S with low penalty
    - m ∈ Int > 0
"""
function update_set!(S, ps, m)
    S_int = findall(S .== 1) # binaries to int, the location where S == 1
    id_remove_S = sortperm(ps[S])[end-(m-1):end] # select last m (m-largest) penalties of S
    id_add = sortperm(ps)[1:m] # get the id which contains the lowest penalties
    # replace the ids:
    S[S_int[id_remove_S]] .= 0
    S[id_add] .= 1
end


"""
initialize opt,u(x),f(x) -> p(x) (u(x),f(x),p(x) table ∀x) given some simulation data tables
should be computed only once per batch
"""
function main_init_opttable()
    # load basic info:
    E = readdlm("data/energies.txt")
    n_data = length(E)
    ids_data = 1:n_data 
    # simulation data tables:
    tb_centers = readdlm("data/custom_CMBDF_centers_181023.txt", Int)[:, 1:100]
    tb_maes = vec(readdlm("result/deltaML/MAE_custom_CMBDF_centers_181023.txt"))
    # transform centers to binaries:
    tb_centers_bin = zeros(Bool, size(tb_centers, 1), n_data)
    for i ∈ axes(tb_centers, 1)
        tb_centers_bin[i, :] = int_to_bin(tb_centers[i,:], n_data)
    end
    # get minima:
    id_opt = argmin(tb_maes)
    opt = tb_maes[id_opt]
    # compute penalty of x:
    p = zeros(n_data); f = zeros(n_data); u = zeros(Int, n_data)
    t = @elapsed begin
        init_penalties_x!(p, f, u, ids_data, opt, tb_maes, tb_centers_bin)
    end
    writedlm("data/tsopt/table_penalties.txt", [p f u])
    writedlm("data/tsopt/opt.txt", opt)
    display([p f u])
    display(t)

    # get the n = num processors top training sets:
    
end

function test_pen()
    id_data = 1:6
    tb_centers = [1 2 3 4; 1 3 5 4; 1 2 5 3]
    tb_maes = [15.; 10.; 20]
    p = zeros(length(id_data)); f = zeros(length(id_data)); u = zeros(Int, length(id_data))
    for i ∈ id_data
        p[i], f[i], u[i] = init_penalty_x(i, f[i], u[i], 10., tb_maes, tb_centers)
    end
    display([p, f, u])
end

function test_update()
    E = readdlm("data/energies.txt")
    n_data = length(E)
    ps = fs = zeros(n_data); us = zeros(Bool, n_data)
    # load penalty infos:
    tbp = readdlm("data/tsopt/table_penalties.txt"); ps = tbp[:, 1]; fs = tbp[:, 2]; us = tbp[:, 3]
    display(ps)
    opt = readdlm("data/tsopt/opt.txt")[1]
    # dummy simulator returns (fobj, S): # turns out simulator doesnt need to return S, since its already recorwded in masterr
    fobj = 10.
    opt = fobj < opt ? fobj : opt # sorting new opt value
    display(opt)
    S = int_to_bin([1,2,10,130800], n_data)
    display([ps[S], fs[S], us[S]])
    i = [4,5,6] # sample non perturbed
    update_penalties_x!(ps, fs, us, fobj, opt, S)
    display([ps[S], fs[S], us[S]])
end

function test_update_set()
    Random.seed!(777)
    E = readdlm("data/energies.txt")
    n_data = length(E)
    ps = fs = zeros(n_data); us = zeros(Bool, n_data)
    # load penalty infos:
    tbp = readdlm("data/tsopt/table_penalties.txt"); ps = tbp[:, 1]; fs = tbp[:, 2]; us = tbp[:, 3]
    display(ps)
    opt = readdlm("data/tsopt/opt.txt")[1]
    # dummy S:
    S_int = sample(1:n_data, 100, replace=false)
    println(S_int)
    S = int_to_bin(S_int, n_data)
    update_set!(S, ps, 10)
    println(findall(S .== 1))
    println(setdiff(findall(S .== 1), S_int))
end


function test_main_master()
    
end