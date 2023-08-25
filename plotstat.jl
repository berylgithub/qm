using Plots, DelimitedFiles, LaTeXStrings, Printf, Statistics

function plot_mae()
    molnames = readdir("result")[2:end]
    count = 1
    m = molnames[1]
    N_set = parse.(Int, readdlm("result/"*m*"/err_$m.txt", '\t', String, '\n')[end-16:end, 2])
    println(N_set)
    for m ∈ molnames
        MAEs = parse.(Float64, readdlm("result/"*m*"/err_$m.txt", '\t', String, '\n')[end-16:end,3])
        sidx = sortperm(MAEs) # sort MAE ascending
        println(sidx, " ", MAEs[sidx], " ", N_set[sidx])
        sNset = string.(N_set[sidx]); sMAEs = MAEs[sidx];
        s = scatter(sNset, sMAEs, xticks = sNset,
                    title = m, xlabel = "N", ylabel = "MAE (kcal/mol)", xrotation = -45, xtickfontsize=6, legend=false)
        display(s)
        savefig(s, "plot/MAE_$m.png")
        count += 1
    end
    #mean runtime:
    T = 0.
    for i ∈ eachindex(molnames)
        m = molnames[i]
        mT = parse.(Float64, readdlm("result/"*m*"/err_$m.txt", '\t', String, '\n')[:, end-2:end-1])
        T += sum(mT)/size(mT)[1]
    end
    println(T/length(molnames))

end

function plot_mae_spec()
    molname = "H7C8N1"
    M_set = parse.(Int, readdlm("result/"*molname*"/err_$molname.txt", '\t', String, '\n')[end-9:end, 1])
    MAE = parse.(Float64, readdlm("result/"*molname*"/err_$molname.txt", '\t', String, '\n')[end-9:end,3])
    MAE_old = parse.(Float64, readdlm("result/"*molname*"/err_$molname.txt", '\t', String, '\n')[17+8:17+17,3])
    display(M_set)
    display(MAE)
    display(MAE_old)
    s = scatter(M_set, [MAE_old, MAE], xticks = M_set, markershape = [:cross :xcross], labels = ["MAD" "farthest-dist"], legend_position = :outertopright,
                title = molname, xlabel = L"$M$", ylabel = "MAE (kcal/mol)", xrotation = -45, xtickfontsize=6)
    display(s)
    savefig(s, "plot/MAE_$molname.png")
end

"""
ΔML stuffs:
"""

"""
very specific function, may be changed at will
query the information from a table of the row with the minimum MAE
"""
function query_min(table; feature_type = "")
    # get the min MAE:
    indices = []
    if !isempty(feature_type)
        for i ∈ axes(table, 1)
            if (table[i, 2] == feature_type)
                push!(indices, i)
            end
        end
        sliced = table[indices,:]
        minid = argmin(sliced[:, 7])
        selid = indices[minid] # assume 100 Ntrain is always the lowest MAE
    else
        selid = argmin(table[:, 7])
    end
    return selid
end

"""
query the row index of data by column info
params:
    - colids = list of column ids
    - coldatas = list of data entry corresponding to the colids
"""
function query_indices(tb, colids, coldatas)
    ids = []
    for i ∈ axes(tb, 1)
        c = 0;
        # loop all column ids:
        for (j,colid) ∈ enumerate(colids)
            if tb[i, colid] == coldatas[j]
                c += 1
            end
        end
        if c == length(colids)
            push!(ids, i)
        end
    end
    return ids
end

"""
automatic yticks generator given the number of desired points, and data points
"""
function yticks_generator(data, n)
    n = n+1
    mind = minimum(data); maxd = maximum(data)
    q25 = quantile(data, .1); q75 = quantile(data, .9); # quantiles
    mid = median(data)
    # generate left range and right range of median:
    nhalf = n ÷ 2;
    # split to 4 parts rather than 2?:
    # left:
    mult = max(10^(ndigits(Int(round(abs(mid - q25)))) - 1), 10)
    rl = range(q25, mid, nhalf); rl = rl .- (rl .% mult)
    # right: 
    mult = max(10^(ndigits(Int(round(abs(q75 - mid)))) - 1), 10)
    rr = range(mid, q75, nhalf); rr = rr .- (rr .% mult)
    # combine:
    yticks = [mind; rl; rr; maxd]
    yticks = yticks[yticks .> 0]
    return yticks
end

"""
plot prototype for delta levels
"""
function plot_MAE_db()
    tb = readdlm("result/deltaML/MAE_enum_ns_dn_PCAdn_020723.txt")
    tbsel = readdlm("result/deltaML/MAE_enum_s2_dn_PCAdn_020723.txt")
    
    # plot prototype for each {feature, model, solver}:
    ind = Dict("acsf" => [13:16, 29:32], "soap" => [41:44, 57:60], "fchl19" => [77:80, 85:88]) # db indexing
    inn = Dict("acsf" => [13:16, 29:32, 29:32, 45:48, 45:48], 
                "soap" => [57:60, 73:76, 73:76, 89:92, 89:92], 
                "fchl19" => [97:100, 113:116, 117:120, 129:132, 133:136]) # indices for dn output

    tb_da = tb[inn["fchl19"][1], :]
    tb_db = tb[inn["fchl19"][2], :]
    tbsel_db = tbsel[inn["fchl19"][3], :]
    tb_dn = tb[inn["fchl19"][4], :]
    tbsel_dn = tbsel[inn["fchl19"][5], :]
    
    MAEs = vcat(tb_da[:, 7], tb_db[:, 7], tb_dn[:, 7], tbsel_db[:, 7], tbsel_dn[:, 7])
    xticks = tb_da[:, 1]
    xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = round.(vcat(maximum(MAEs), minimum(MAEs)), digits=3)
    yticks = vcat(yticks, range(10, 50, 5))
    ytformat = vcat(string.(yticks[1:2]), map(x -> @sprintf("%.0f",x), yticks[3:end]))
    p = plot(tb_da[:, 1], [tb_da[:, 7], tb_db[:, 7], tbsel_db[:, 7], tb_dn[:, 7], tbsel_dn[:, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:xcross :cross :rect :auto :auto], markersize = [6 6 4 5 5],
        labels = ["MAE(da)" "MAE(db)" "MAE(db,sel)" "MAE(dn)" "MAE(dn,sel)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    savefig(p, "plot/deltaML/MAE_PCAdn_FCHL19_best.png")

    # plot prototype for vs{features} in the same environment:
    tb1 = tbsel[29:32, :] #ACSF
    tb2 = tbsel[61:64, :] #SOAP
    tb3 = tbsel[93:96, :] #FCHL19
    MAEs = vcat(tb1[:, 7], tb2[:, 7], tb3[:, 7])
    xticks = tb1[:, 1]
    xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = round.(vcat(maximum(MAEs), minimum(MAEs)), digits=3)
    yticks = vcat(yticks, range(10, 50, 5))
    ytformat = vcat(string.(yticks[1:2]), map(x -> @sprintf("%.0f",x), yticks[3:end]))
    p = plot(tb1[:, 1], [tb1[:, 7], tb2[:, 7], tb3[:, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:xcross :cross :rect], markersize = [6 6 4],
        labels = ["MAE(ACSF)" "MAE(SOAP)" "MAE(FCHL19)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    #savefig(p, "plot/deltaML/MAE_features_vs.png")
end

function plot_MAE_dt()
    tbns = readdlm("result/deltaML/MAE_enum_ns_dt_180823.txt")
    tbs = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt")
    tb = tbs # select which table
    # see effect of dressed on each feature:
    d_minind = Dict("acsf" => query_min(tb, feature_type = "ACSF_51"), 
                "soap" => query_min(tb, feature_type = "SOAP"), 
                "fchl19" => query_min(tb, feature_type = "FCHL19"))
    d_colq = Dict("acsf" =>tb[d_minind["acsf"],[2,3,4]], 
                "soap" =>tb[d_minind["soap"],[2,3,4]], 
                "fchl19"=>tb[d_minind["fchl19"],[2,3,4]])
    d_ind = Dict("acsf"=>query_indices(tb, [2,3,4], d_colq["acsf"]),
                "soap"=>query_indices(tb, [2,3,4], d_colq["soap"]), 
                "fchl19"=>query_indices(tb, [2,3,4], d_colq["fchl19"]))
    # observe per feature:
    # ind = d_ind["acsf"] # example of selecting column by feature type

    # 1) fix the best hyperparameter from ns and sx data then compare up to dt (see the effect of baselines):
    tables = [tbns, tbs] # noselect and select
    tbnames = ["ns", "s2"]
    for (i,tb) ∈ enumerate(tables)
        minid = query_min(tb)
        qcol = tb[minid, [2,3,4]]
        ind = query_indices(tb, [2,3,4], qcol)
        MAEs = tb[ind, end]
        xticks = tb[:, 1][1:4]
        xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
        y = tb[ind, 7]
        yticks = yticks_generator(y, 5)
        ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
        p = plot(xticks, [tb[ind, :][1:4, end], tb[ind, :][5:8, end], tb[ind, :][9:12, end], tb[ind, :][end-3:end, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["MAE(da)" "MAE(db)" "MAE(dn)" "MAE(dt)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
        display(p)
        println(join(tb[ind, [2,3,4]][1,:], "-"))
        savefig(p, "plot/deltaML/MAE_bh-"*join(tb[ind, [2,3,4]][1,:], "-")*"_"*tbnames[i]*"_upto-dt.png")
    end

    # 2) - get the best from ns and s then fix hyperparam then plot for each ns and s,
    #   - get each best of ns and s
    # (4 curves total, see the efefect of data selection) 
    # fix the hyperparameters in which the best from both ns and s:
    jointb = vcat(tbns, tbs)
    minid = query_min(jointb)
    qcol = jointb[minid, [2,3,4,5]]
    id_ns = query_indices(tbns, [2,3,4,5], qcol)
    id_s = query_indices(tbs, [2,3,4,5], qcol)
    # best of ns mode:
    minid = query_min(tbns)
    id_bns = query_indices(tbns, [2,3,4,5], tbns[minid, [2,3,4,5]])
    jointb = vcat(tbns[id_ns, :], tbns[id_bns, :], tbs[id_s, :])

    xticks = jointb[:, 1][1:4]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    MAEs = jointb[:, end]; minMAE = minimum(MAEs); maxMAE = maximum(MAEs);
    yticks = range(minMAE, maxMAE, 7)
    y = sort(vcat(tbns[id_ns, 7], tbns[id_bns, 7], tbs[id_s, 7]))
    yticks = yticks_generator(y, 5)
    #= yticks = yticks[2:end-1] .- (yticks[2:end-1] .% 10) # round with 10 as multiplier
    yticks = yticks[yticks .> 0.] # remove zeros
    yticks = vcat(minMAE, yticks, maxMAE) # concat with min and max =#
    ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, [tbns[id_ns, 7], tbns[id_bns, 7], tbs[id_s, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
        labels = ["MAE(ns)" "MAE(bons)" "MAE(bos)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    savefig(p, "plot/deltaML/MAE_sel_effect.png")

    # 3) fix the best MAE on each feature (1 plot w/ 3 curves, see the effect of feature)
    ftypes = ["ACSF_51", "SOAP", "FCHL19"]
    jtb = vcat(tbns, tbs)
    nrow = size(jtb, 1); halfrow = (nrow ÷ 2); sid_start = halfrow + 1; # since no table starting identifier
    tbslices = [] # reference to slices of tables
    # find min location (in which location of table):
    minids = []
    for (i, ftype) ∈ enumerate(ftypes)
        minid = query_min(jtb; feature_type = ftype)
        push!(minids, minid)
        if minid > halfrow # get from s2 table:
            minid = minid - halfrow
            qid = query_indices(tbs, [2,3,4,5], tbs[minid, [2,3,4,5]])
            push!(tbslices, tbs[qid, :]) 
        else
            qid = query_indices(tbns, [2,3,4,5], tbns[minid, [2,3,4,5]])
            push!(tbslices, tbns[qid, :])
        end
    end
    display(tbslices)

end