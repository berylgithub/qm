using DelimitedFiles, LaTeXStrings, Printf, Statistics
using Plots
using PlotlyJS 
const jplot = PlotlyJS.plot
const jscatter = PlotlyJS.scatter
const jsavefig = PlotlyJS.savefig


using Graphs, MolecularGraph, Luxor, Images # for visualization
using Rotations, AngleBetweenVectors
#using Polynomials
using MathTeXEngine

include("utils.jl")

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
    return unique(yticks)
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
    d_minind = Dict("acsf" => query_min_f(tb, feature_type = "ACSF_51"), 
                "soap" => query_min_f(tb, feature_type = "SOAP"), 
                "fchl19" => query_min_f(tb, feature_type = "FCHL19"))
    d_colq = Dict("acsf" =>tb[d_minind["acsf"],[2,3,4]], 
                "soap" =>tb[d_minind["soap"],[2,3,4]], 
                "fchl19"=>tb[d_minind["fchl19"],[2,3,4]])
    d_ind = Dict("acsf"=>query_indices(tb, [2,3,4], d_colq["acsf"]),
                "soap"=>query_indices(tb, [2,3,4], d_colq["soap"]), 
                "fchl19"=>query_indices(tb, [2,3,4], d_colq["fchl19"]))


    # 1) fix the best hyperparameter from ns and sx data then compare up to dt (see the effect of baselines) (2 plots):
    tables = [tbns, tbs] # noselect and select
    tbnames = ["ns", "s2"]
    for (i,tb) ∈ enumerate(tables)
        minid = query_min_f(tb)
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
        writedlm("plot/deltaML/MAE_bh-"*join(tb[ind, [2,3,4]][1,:], "-")*"_"*tbnames[i]*"_upto-dt.txt", [tb[ind, :][1:4, :]; tb[ind, :][5:8, :]; tb[ind, :][9:12, :]; tb[ind, :][end-3:end, :]]) # store the plot information
    end

    # 2) - get the best from ns and s then fix hyperparam then plot for each ns and s,
    #   - get each best of ns and s
    # (1 plot, 4 curves total, see the efefect of data selection) 
    # fix the hyperparameters in which the best from both ns and s:
    jointb = vcat(tbns, tbs)
    minid = query_min_f(jointb)
    qcol = jointb[minid, [2,3,4,5]]
    id_ns = query_indices(tbns, [2,3,4,5], qcol)
    id_s = query_indices(tbs, [2,3,4,5], qcol)
    # best of ns mode:
    minid = query_min_f(tbns)
    id_bns = query_indices(tbns, [2,3,4,5], tbns[minid, [2,3,4,5]])
    jointb = vcat(tbns[id_ns, :], tbns[id_bns, :], tbs[id_s, :])

    xticks = jointb[:, 1][1:4]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    MAEs = jointb[:, end]; minMAE = minimum(MAEs); maxMAE = maximum(MAEs);
    yticks = range(minMAE, maxMAE, 7)
    y = sort(vcat(tbns[id_ns, 7], tbns[id_bns, 7], tbs[id_s, 7]))
    yticks = yticks_generator(y, 5)
    ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, [tbns[id_ns, 7], tbns[id_bns, 7], tbs[id_s, 7]],
        yticks = (yticks, ytformat), xticks = (xticks, xtformat),
        xaxis = :log, yaxis = :log,
        markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
        labels = ["MAE(ns)" "MAE(bons)" "MAE(bos)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)")
    display(p)
    savefig(p, "plot/deltaML/MAE_sel_effect.png")
    writedlm("plot/deltaML/MAE_sel_effect.txt", [tbns[id_ns, :]; tbns[id_bns, :]; tbs[id_s, :]])

    # 3) fix the best MAE on each feature (1 plot w/ 3 curves, see the effect of feature)
    ftypes = ["ACSF_51", "SOAP", "FCHL19"]
    jtb = vcat(tbns, tbs)
    nrow = size(jtb, 1); halfrow = (nrow ÷ 2);
    tbslices = [] # reference to slices of tables
    # find min location (in which location of table):
    minids = []
    for (i, ftype) ∈ enumerate(ftypes)
        minid = query_min_f(jtb; feature_type = ftype)
        push!(minids, minid)
        println([ftype, minid, halfrow])
        if minid > halfrow # get from s2 table:
            minid = minid - halfrow
            qid = query_indices(tbs, [2,3,4,5], tbs[minid, [2,3,4,5]])
            push!(tbslices, tbs[qid, :]) 
        else
            qid = query_indices(tbns, [2,3,4,5], tbns[minid, [2,3,4,5]])
            push!(tbslices, tbns[qid, :])
        end
    end
    xticks = tbs[1:4, 1]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    jtb = reduce(vcat, tbslices)
    MAEs = jtb[:, end]
    yticks = yticks_generator(MAEs, 5); ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, [tbslices[1][:, end], tbslices[2][:, end], tbslices[3][:, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["MAE(acsf)" "MAE(soap)" "MAE(fchl19)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    savefig(p, "plot/deltaML/MAE_best_each-feature.png")
    writedlm("plot/deltaML/MAE_best_each-feature.txt", jtb)
end

function plot_subsec_PCA()
    #4) for PCA subsection, 1 plot 4 curves, each curve is the best of each elvl on s2 data (see how far PCA can improve things)
    tb1 = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt") # wait for the computation
    tb2 = readdlm("result/deltaML/MAE_enum_s2_dt_PCAjl-dn5-dt5-280823.txt")
    # best without PCA:
    minid = query_min(tb1, [], [])
    id_minnopca = query_indices(tb1, [2,3,4,5], tb1[minid, [2,3,4,5]])
    display(tb1[id_minnopca, :])
    # best of DN without PCA:
    minid = query_min(tb1, [5], ["dressed_angle"])
    id_mindn = query_indices(tb1, [2,3,4,5], tb1[minid, [2,3,4,5]])
    display(tb1[id_mindn, :])
    # best of DT without PCA:
    minid = query_min(tb1, [5], ["dressed_torsion"])
    id_mindt = query_indices(tb1, [2,3,4,5], tb1[minid, [2,3,4,5]])
    display(tb1[id_mindt, :])
    # best of DN with PCA:
    minid = query_min(tb2, [5], ["dressed_angle"])
    id_mindn_pca = query_indices(tb2, [2,3,4,5], tb2[minid, [2,3,4,5]])
    display(tb2[id_mindn_pca, :])
    # best of DT with PCA:
    minid = query_min(tb2, [5], ["dressed_torsion"])
    id_mindt_pca = query_indices(tb2, [2,3,4,5], tb2[minid, [2,3,4,5]])
    display(tb2[id_mindt_pca, :])
    # get all selected MAEs ....:
    MAEs = vcat(tb1[id_minnopca, end], tb1[id_mindn, end], tb1[id_mindt, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end])
    # generate the ticks:
    xticks = tb1[1:4, 1]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = yticks_generator(MAEs, 7); ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    # plot:
    p = plot(xticks, [tb1[id_minnopca, end], tb1[id_mindn, end], tb1[id_mindt, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle :pentagon], markersize = (ones(5)*6)',
            labels = ["MAE(bo,db)" "MAE(bo,dn)" "MAE(bo,dt)" "MAE(bo,dn,PCA1)" "MAE(bo,dt,PCA1)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    savefig(p, "plot/deltaML/MAE_PCA1_comparison.png")
    writedlm("plot/deltaML/MAE_PCA1_comparison.txt", [tb1[id_minnopca, :]; tb1[id_mindn, :]; tb1[id_mindt, :]; tb2[id_mindn_pca, :]; tb2[id_mindt_pca, :]])

    # 5) (6 curves) BODB (from above), PCA1BODN (from above), PCA1BODT (from above), PCA2BODB, PCA2BODN, PCA2BODT:
    tb3 = readdlm("result/deltaML/MAE_enum_s2_dt_PCAjl-db5-dn5-dt5-300823.txt")
    # PCA2 BODB:
    minid = query_min(tb3, [5], ["dressed_bond"])
    id_mindb2 = query_indices(tb3, [2,3,4,5], tb3[minid, [2,3,4,5]])
    display(tb3[id_mindb2, :])
    # PCA2 BODN:
    minid = query_min(tb3, [5], ["dressed_angle"])
    id_mindn2 = query_indices(tb3, [2,3,4,5], tb3[minid, [2,3,4,5]])
    display(tb3[id_mindn2, :])
    # PCA2 BODT:
    minid = query_min(tb3, [5], ["dressed_torsion"])
    id_mindt2 = query_indices(tb3, [2,3,4,5], tb3[minid, [2,3,4,5]])
    display(tb3[id_mindt2, :])
    MAEs = vcat(tb1[id_minnopca, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end], tb3[id_mindb2, end], tb3[id_mindn2, end], tb3[id_mindt2, end])
    yticks = yticks_generator(MAEs, 7); ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    # plot:
    p = plot(xticks, [tb1[id_minnopca, end], tb2[id_mindn_pca, end], tb2[id_mindt_pca, end], tb3[id_mindb2, end], tb3[id_mindn2, end], tb3[id_mindt2, end]],
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle :pentagon :star], markersize = (ones(5)*6)',
            labels = ["MAE(bo,db)" "MAE(bo,dn,PCA1)" "MAE(bo,dt,PCA1)" "MAE(bo,db,PCA2)" "MAE(bo,dn,PCA2)" "MAE(bo,dt,PCA2)"], 
            linestyles = [:solid :solid :solid :dash :dash :dash],
            xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    savefig(p, "plot/deltaML/MAE_PCA_1_2_comparison.png")
    writedlm("plot/deltaML/MAE_PCA_1_2_comparison.txt",[tb1[id_minnopca, :], tb2[id_mindn_pca, :], tb2[id_mindt_pca, :], tb3[id_mindb2, :], tb3[id_mindn2, :], tb3[id_mindt2, :]])

    # 4+5) Table of best of non PCA, best of PCA1 for each level, best of PCA2 for each level:
    # best of DN and DT non PCA:
    minid = query_min(tb1, [5], ["dressed_angle"])
    jtb = [reshape(tb1[id_minnopca[end], :], 1, :); reshape(tb1[id_mindn[end], :], 1, :); reshape(tb1[id_mindt[end], :], 1, :);
            reshape(tb2[id_mindn_pca[end], :], 1, :); reshape(tb2[id_mindt_pca[end], :], 1, :); 
            reshape(tb3[id_mindb2[end], :], 1, :); reshape(tb3[id_mindn2[end], :], 1, :); reshape(tb3[id_mindt2[end], :], 1, :)] 
    jtb = hcat(jtb, ["none", "none", "none", "PCA1", "PCA1", "PCA2", "PCA2", "PCA2"])
    # process (4+5) table:
    # move last column to first:
    jtb[:, 1] = jtb[:, end]
    jtb = jtb[:, 1:end-1]
    jtb[:, 5] = ["DB", "DN", "DT", "DN", "DT", "DB", "DN", "DT"]
    for i ∈ axes(jtb, 1)
        # change featurenames:
        if jtb[i, 2] == "ACSF_51"
            jtb[i, 2] = "ACSF"
        end
        # change model names:
        if jtb[i, 3] == "REAPER"
            jtb[i, 3] = "DPK"
        elseif jtb[i, 3] == "GAK"
            jtb[i, 3] = "GK"
        end
    end
    jtb = jtb[:, vcat(1:end-2, end)]
    display(jtb)
    jtb[:, end] = clean_float(jtb[:, end])
    writelatextable(jtb, "plot/deltaML/tb_PCA_comparison.tex")
    
end


function main_write_full_tbs()
    tb1 = readdlm("result/deltaML/MAE_enum_ns_dt_180823.txt")
    tb2 = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt")
    # query DN and DT from each:
    id_tb1_dn = query_indices(tb1, [5], ["dressed_angle"])
    id_tb1_dt = query_indices(tb1, [5], ["dressed_torsion"])
    id_tb2_dn = query_indices(tb2, [5], ["dressed_angle"])
    id_tb2_dt = query_indices(tb2, [5], ["dressed_torsion"])
    jtb1 = [tb1[id_tb1_dn, :]; tb1[id_tb1_dt, :]] 
    jtb2 = [tb2[id_tb2_dn, :]; tb2[id_tb2_dt, :]]
    jtbs = [jtb1, jtb2]
    for jtb ∈ jtbs # should just make a mandatory function for these guys
        for i ∈ axes(jtb, 1)
            # change featurenames:
            if jtb[i, 2] == "ACSF_51"
                jtb[i, 2] = "ACSF"
            end
            # change model names:
            if jtb[i, 3] == "REAPER"
                jtb[i, 3] = "DPK"
            elseif jtb[i, 3] == "GAK"
                jtb[i, 3] = "GK"
            end
            # change elevel names
            if jtb[i, 5] == "dressed_angle"
                jtb[i, 5] = "DN"
            elseif jtb[i, 5] == "dressed_torsion"
                jtb[i, 5] = "DT"
            end
        end
        jtb[:, [end-1, end]] = clean_float(jtb[:, [end-1, end]])
    end
    writelatextable(jtbs[1], "result/deltaML/tb_ns_DNDT.tex")
    writelatextable(jtbs[2], "result/deltaML/tb_s2_DNDT.tex")
end

function main_tb_hda()
    tbbase1 = readdlm("result/deltaML/MAE_base_s2_dt_180823.txt")
    tb1 = readdlm("result/deltaML/MAE_enum_s2_dt_180823.txt")
    tbbase2 = readdlm("result/deltaML/MAE_base_s2_hda_dt_120923.txt")
    tb2 = readdlm("result/deltaML/MAE_enum_s2_hda_dt_120923.txt")
    
    # base MAE table
    tbb1x = replace(tbbase1, "dressed_atom" => "DA", "dressed_bond" => "DB", 
                    "dressed_angle" => "DN", "dressed_torsion" => "DT")
    tbb2x = replace(tbbase2, "dressed_atom" => "DA", "dressed_bond" => "DB", 
                    "dressed_angle" => "DN", "dressed_torsion" => "DT")
    jtb = hcat(tbb1x, tbb2x[:, end-1:end])
    jtb[2:end, 2:end] = clean_float(jtb[2:end, 2:end])
    writelatextable(jtb, "result/deltaML/tb_base_vs_hda.tex", hline=false)
    # enum table comparison (DA vs HDA best on DA level, global minimum DA vs global minimum HDA):
    id1da = query_min(tb1, [5], ["dressed_atom"]) # best of DA
    id2da = query_min(tb2, [5], ["dressed_atom"])
    id1min = query_min(tb1, [], [])
    id2min = query_min(tb2, [], [])
    jtb = vcat(reshape(tb1[id1da, :], 1, :), reshape(tb2[id2da, :], 1, :), reshape(tb1[id1min, :], 1, :), reshape(tb2[id2min, :], 1, :))
    jtb = hcat(jtb, ["standard", "hybridized", "standard", "hybridized"])
    writedlm("result/deltaML/tb_enum_da-vs-hda.txt", jtb)
    jtb_tex = Matrix{Any}(undef, 3, 3)
    jtb_tex[1, :] = ["query", "SDA MAE test", "HDA MAE test"]
    jtb_tex[2, :] = vcat("best on DA", jtb[1:2, end-1])
    jtb_tex[3, :] = vcat("best from all", jtb[3:4, end-1])
    jtb_tex[2:end, 2:end] = clean_float(jtb_tex[2:end, 2:end])
    writelatextable(jtb_tex, "result/deltaML/tb_enum_da-vs-hda.tex"; hline=false)
end

function MAE_enum_v2_plot()
    tb = readdlm("result/deltaML/MAE_enum_v2_30k_100k_H_280923.txt") # assume the default includes Hydrogens
    display(tb)
    minid = query_min(tb, [], [], 9) # 9 is the index of the test MAE
    # query b_MAEs_* from one table:
    b_MAEs = [] # will be a vector of vectors
    bs = ["A", "AB", "ABN", "ABNT"]
    for (i,b) ∈ enumerate(bs)
        qid = query_indices(tb, [3, 4, 5], [b, "GK", "ACSF_51"])
        push!(b_MAEs, tb[qid, 7])
    end
    xticks = tb[1:8, 1]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    yticks = yticks_generator(reduce(vcat, b_MAEs), 7) ∪ [12, 16];
    ytformat = map(x -> @sprintf("%.0f",x), yticks)
    #ytformat = vcat(string(round(yticks[1], digits=3)), map(x -> @sprintf("%.0f",x), yticks[2:end-1]), string(round(yticks[end], digits=3)))
    p = plot(xticks, b_MAEs,
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["A" "AB" "ABN" "ABNT"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    # sample w/ for (elv, solver) = ([A, AB, ABN, ABNT], GK):
    MAEs = []
    for (i,b) ∈ enumerate(bs)
        qid = query_indices(tb, [3, 4, 5], [b, "GK", "FCHL19"])
        push!(MAEs, tb[qid, 9])
    end
    qids_A = query_indices(tb, [3, 4,5], ["A", "GK", "FCHL19"])
    qids_AB = query_indices(tb, [3, 4,5], ["AB", "GK", "FCHL19"])
    # e.g., on FCHL19 space:
    ys = [tb[qids_A, 9], tb[qids_AB, 9]]
    yticks = [2.0^i for i ∈ 0:4]
    ytformat = map(x -> @sprintf("%.0f",x), yticks)
    p = plot(xticks, MAEs,
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["A" "AB" "ABN" "ABNT"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    hline!([1], labels = nothing)
    display(p)
    # look at MBDF acc to determine whether it's feasible for dataselection:
    tb2 = readdlm("result/deltaML/MAE_enum_v2_MBDF_30k_100k_H_280923-2.txt")
    minid2 = query_min(tb2, [], [], 9)
    display(tb2[minid2,9])
    qids_A = query_indices(tb, [3,4,5], ["A", "GK", "ACSF_51"])
    qids_AB = query_indices(tb, [3,4,5], ["AB", "GK", "ACSF_51"])
    qids1_A = query_indices(tb, [3,4,5], ["A", "GK", "FCHL19"])
    qids1_AB = query_indices(tb, [3,4,5], ["AB", "GK", "FCHL19"])
    qids2_A = query_indices(tb2, [3,4,5], ["A", "GK", "MBDF"])
    qids2_AB = query_indices(tb2, [3,4,5], ["AB", "GK", "MBDF"])
    ys = [tb[qids_A, 9], tb[qids_AB, 9], tb[qids1_A, 9], tb[qids1_AB, 9], tb2[qids2_A, 9], tb2[qids2_AB, 9]]
    p = plot(xticks, ys,
            yticks = (yticks, ytformat), xticks = (xticks, xtformat),
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :diamond :utriangle], markersize = (ones(5)*6)',
            labels = ["(ACSF, A)" "(ACSF, AB)" "(FCHL19, A)" "(FCHL19, AB)" "(MBDF, A)" "(MBDF, AB)"], xlabel = "Ntrain", ylabel = "MAE (kcal/mol)"
        )
    display(p)
    qid100 = [] 
    display(tb2[qids2_A,:])
end

function test_bond_energy_plot()
    # ZiZkr^-c:
    function pairpot(Z1, Z2, r, c)
        return Z1*Z2/r^c
    end
    Zs = Float64.([1,6,7,8,9]) # [H, C, N, O, F]
    rs = range(0., 5., 1000)
    cs = Float64.([1, 6])
    
    x = rs
    y = pairpot.(Zs[2], Zs[4], rs, cs[2])
    display(y)
    plot(x, y, ylims = [0, 100])

    # morse:
    function morse_pot(r, D, a, r0, s; mode=1)
        if mode==1
            return D*(exp(-2*a*(r-r0)) - 2*exp(-a*(r-r0))) + s # additional shift constant s for "pure" fitting
        elseif mode == 2
            return D*(1-exp(-a*(r-r0)))^2
        end
    end
    D = .5
    a = 2.
    r0 = 1.
    s = 0.1
    x = rs
    y = morse_pot.(rs, D, a, r0, s,mode=1)
    display(y)
    plot(x, y, ylims = [-1., 1.])
end


"""
combine rand useq table by adding new column (H and HDA too) 
"""
function main_combine_tbs()
    headers = ["ntrain", "ntest", "elv", "model", "feature", 
                "b_MAEtrain", "b_MAEtest", "MAEtrain", "MAEtest", 
                "t_mcom", "t_etrain", "t_epred", "t_mtrain", "t_mtest"]
    headers_add = ["select", "hydrogen","hybrid"] # additional columns to be combined
    headers = vcat(headers_add, headers)
    display(headers)
    # combine tables:
    tb1 = readdlm("result/deltaML/MAE_enum_v2_30k_100k_srand_H_101123.txt")
    tb2 = readdlm("result/deltaML/MAE_enum_v2_30k_100k_sid57_H_101123.txt")
    tb3 = readdlm("result/deltaML/MAE_enum_v2_30k_100k_sid57_H_HDA_101123.txt") # later when the computation is finished
    tbj = Matrix{Any}(undef, sum(size.([tb1, tb2, tb3], 1))+1, size(tb1, 2)+length(headers_add))
    tbj[1,:] = headers
    id_tbs = [(2,1+size(tb1,1)), (2+size(tb1,1), size(tb1,1)+size(tb2,1)+1), (size(tb1,1)+size(tb2,1)+2, size(tb1,1)+size(tb2,1)+size(tb3,1)+1)]
    display(id_tbs)
    tbj[id_tbs[1][1]:id_tbs[1][2],1:3] .= ["rand" "true" "false"]; tbj[id_tbs[1][1]:id_tbs[1][2],4:end] = tb1
    tbj[id_tbs[2][1]:id_tbs[2][2],1:3] .= ["sid57" "true" "false"]; tbj[id_tbs[2][1]:id_tbs[2][2],4:end] = tb2
    tbj[id_tbs[3][1]:id_tbs[3][2],1:3] .= ["sid57" "true" "true"]; tbj[id_tbs[3][1]:id_tbs[3][2],4:end] = tb3
    writedlm("result/deltaML/MAE_enum_v2_combined_101123.txt", tbj)
end


"""
several plotting scenarios:
- plot of the minimum of each ntrain separated by each feature -> 6 curves
- plot of rand vs usequence, sample ACSF, FCHL19, and CMBDF (6 curves, 2 each representation)
- plot ΔML, sample ACSF, FCHL19, and CMBDF -> (3 plots * 4 curves, reduce if too crowded)
"""
function main_plot_v2()
    tb = readdlm("result/deltaML/MAE_enum_v2_combined_101123.txt")
    headers = tb[1,:] # headers
    println(collect(enumerate(headers)))
    tb = tb[2:end, :] # extract values
    id_gmin = query_min(tb, [], [], 12) # global minima 
    
    # plot of minimum on each ntrain on each feature:
    ntrains = tb[1:9,4]
    fts = unique(tb[:,8])
    ids_mins = []
    for ft ∈ fts
        temp = []
        for ntrain ∈ ntrains
            id_min = query_min(tb, [8, 4], [ft, ntrain], 12)
            push!(temp, id_min)
        end
        push!(ids_mins, temp)
    end
    xticks = ntrains; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    ys = [tb[ids_mins[i],12] for i ∈ eachindex(fts)]
    yticks = [2.0^i for i ∈ 0:3]; ytformat = map(x -> @sprintf("%.0f",x), yticks)
    p = plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            markershape = [:circle :rect :utriangle :diamond :hexagon], markersize = (ones(length(fts))*6)',
            labels = "min(" .* permutedims(fts) .*")", xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "minimum MAE for each representation",
            dpi = 1000
        )
    hline!([1], labels = nothing, lc = :red)
    display(p)
    savefig(p, "plot/deltaML/fig_min_MAE_representation.png")
    tbplot = vcat(permutedims(headers),tb[reduce(vcat, ids_mins),:])
    display(tbplot)
    writedlm("plot/deltaML/tb_min_MAE_representation.txt", tbplot)

    # plot min(rand) vs min(usequence), sample (ACSF, FCHL19, CMBDF)
    fts = ["ACSF_51", "FCHL19", "SOAP", "CMBDF"]
    selections = unique(tb[:,1])
    iters = collect(Iterators.product(ntrains, fts, selections)) # iterators, so that the loop is single
    id_mins = []
    for it ∈ iters
        push!(id_mins, query_min(tb, [4, 8, 1], it, 12))
    end
    niters = reduce(*, length.([fts,selections]))
    nslice = Int(length(id_mins)/niters)
    ncurves = Int(length(id_mins)/nslice) # number of curves
    id_slices = [((i-1)*nslice + 1, i*nslice) for i ∈ 1:ncurves] # which slices to pick from the id_mins
    ys = [tb[id_mins[id[1]:id[2]], 12] for id ∈ id_slices] # slice the MAEs (y axis)
    yticks = [2.0^i for i ∈ 0:5]; ytformat = map(x -> @sprintf("%.0f",x), yticks)
    labels = permutedims([tb[id_mins[id[1]],1] .* ", " .* tb[id_mins[id[1]],8] for id ∈ id_slices])
    p = plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            linestyle = [:dash :dash :dash :dash :solid :solid :solid :solid],
            markershape = [:circle :rect :utriangle :diamond :circle :rect :utriangle :diamond], 
            markercolor = [:black :green :blue :purple :black :green :blue :purple],
            linecolor = [:black :green :blue :purple :black :green :blue :purple],
            markersize = (ones(ncurves)*6)',
            labels = labels, xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "Random vs Usequence",
            dpi = 1000
        )
    hline!([1], labels = nothing, lc = :red)
    display(p)
    savefig(p, "plot/deltaML/fig_RandVFPS.png")
    tbplot = vcat(permutedims(headers),tb[reduce(vcat, id_mins),:])
    writedlm("plot/deltaML/tb_RandVFPS.txt", tbplot)

    # plot of the baseline MAE only (4 elvs * 2 selections + 4 sid_57 hybrid = 12 curves)
    # ! looks like this pattern of plotting is repeating, should probably try to write this in 
    elvs = ["A", "AB", "ABN", "ABNT"]
    selections = unique(tb[2:end, 1])
    hybrids = [false, true]
    iters = collect(Iterators.product(ntrains, elvs, selections, hybrids))
    ids_plot = []
    for iter ∈ iters
        ids = query_indices(tb, [4,6,1,3], iter) # only need one data point
        if !isempty(ids) # some of the combination may not exist
            push!(ids_plot, ids[1])
        end
    end
    display(tb[ids_plot,[4,6,1,3]])
    nslice = length(ntrains) # number of x data points each curve
    ncurves = Int(length(ids_plot)/length(ntrains))
    id_slices = [((i-1)*nslice + 1, i*nslice) for i ∈ 1:ncurves]
    ys = [tb[ids_plot[id[1]:id[2]], 10] for id ∈ id_slices] # slice the MAEs (y axis)
    yticks = vcat([2.0^i for i ∈ 0:5], [10., 12., 20.]); ytformat = map(x -> @sprintf("%.0f",x), yticks)
    labels = permutedims([tb[ids_plot[id[1]],6] .* ", " .* tb[ids_plot[id[1]],1] .* ", " .* string(tb[ids_plot[id[1]],3]) for id ∈ id_slices])
    p = plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            linestyle = permutedims(vcat(repeat([:dash],4), repeat([:solid],8))),
            markershape = hcat(permutedims(hcat(repeat([:circle, :rect, :utriangle, :diamond], 2))), [:+ :x :hexagon :heptagon] ), 
            markercolor = permutedims(hcat(repeat([:black, :green, :blue, :purple],3))),
            linecolor = permutedims(hcat(repeat([:black, :green, :blue, :purple],3))),
            markersize = (ones(ncurves)*4)',
            labels = labels, xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "Baseline MAEs",
            legendfontsize = 7, legend = :outertopright,
            dpi=1000
        )
    display(p)
    savefig(p, "plot/deltaML/fig_base.png")
    tbplot = vcat(permutedims(headers),tb[ids_plot,:])
    writedlm("plot/deltaML/tb_base.txt", tbplot)
    
end

function main_get_stats_deltaML()
    # get the hit counts of: [elvs, sid57, hybrid]
    tb = readdlm("result/deltaML/MAE_enum_v2_combined_101123.txt")
    headers = tb[1,:]
    tb = tb[2:end,:]
    println([(k,v) for (k,v) ∈ enumerate(headers)])
    hit_counts = Dict()
    hit_counts 
    #...
end

function main_display_selected_mol()
    trainset = vec(Int.(readdlm("data/tsopt/opt_tracker_freeze.txt")[2:end]))
    tf = map(d->d["formula"],dataset[trainset])
    # generate the string matrix:
    ## subscript all numbers in the formula:
    for (i,data) in enumerate(tf)
        temp = ""
        for chr in data
            if isdigit(chr)
                temp*="\$_"*chr*"\$"
            else
                temp*=chr
            end
        end 
        tf[i] = temp
    end
    display(tf)
    ## write the training set into 10x10 matrix 
    A = Matrix{Any}(undef, 25, 4)
    for i ∈ eachindex(A)
        #A[i] = "\$\\#"*string(trainset[i])*"\$   "*tf[i]
        A[i] = tf[i]
    end
    writelatextable(A, "result/table_selmol.tex"; hline=false)
end

"""
display the deltaML relevancy table
"""
function main_disp_delta_best()
    tb = readdlm("result/deltaML/MAE_enum_sbest_dt_H_220124.txt")
    display(tb)
    # query minimum values foreach feature and dressed fragment:
    features = ["ACSF_51", "SOAP", "FCHL19", "MBDF", "CMBDF", "CM", "BOB"]
    dresses = unique(tb[:,5])
    iters = Iterators.product(dresses, features)
    
    # generate string matrix:
    A = Matrix{Any}(undef, length(dresses)+1, length(features)+1)
    A[1,1] = ""
    A[2:end, 1] = ["DA", "DB", "DN", "DH"]
    A[1, 2:end] = features; A[1,2] = "ACSF"
    display(A) 

    # fill table value:
    B = zeros(4,7)
    for (i,it) ∈ enumerate(iters)
        minid = query_min(tb, [1, 2,5], [100, it[2], it[1]], 7)
        B[i] = tb[minid, 7]
    end
    A[2:end, 2:end] = clean_float.(B)
    display(A)
    writelatextable(A, "result/table_dressed_feature.tex")
end


"""
for visualization of the Kernel feature
"""
function main_PCA_plot()
    # load data:
    E = vec(readdlm("data/energies.txt"))
    K = readdlm("result/deltaML/PCA_kernel_2.txt")
    ev = readdlm("result/deltaML/PCA_eigenvalue_2.txt")
    trains = Int.(vec(readdlm("data/tsopt/opt_tracker_freeze.txt")[2:end]))
    tests = setdiff(1:length(E), trains)
    display(ev)
    # training set display:
    p = Plots.scatter(K[tests,1], K[tests,2], markercolor=:blue, markersize = 3.5, labels = "test", legend = :outertopleft, xlabel = "PC1", ylabel="PC2")
    Plots.scatter!(K[trains,1], K[trains,2], markershape=:utriangle, markercolor=:red, markersize = 6, labels = "train")
    display(p)
    #savefig(p, "plot/deltaML/PCA_kernel_plot.png")

    ## finding pattern shenanigans:
    # display in label instead of marker:
    p1 = Plots.scatter(K[:,1], K[:,2], xlimits = (-0.1, 1.2), ylimits = (-0.1, 1.2), markercolor=:blue, markersize = 3, labels = "molecule", legend = :outertopleft, xlabel = "PC1", ylabel="PC2", dpi=1000)
    #annotate!(0.2, 0.1, text("A", 0.1, :red, :top))
    display(p1)
    Ktrain = K[trains,:]
    Ksel = Ktrain[findall( 0.3 .< Ktrain[:,1] .< 0.7 ),:] # selected K (from visual judgement)
    jp = jplot(jscatter(x=K[:,1], y=K[:,2], mode="markers")) # display using PlotlyJS
    display(jp)
    # linear polynomial fit, sample 2 lines (points obtained manually from checking):
    l1p = [[0.31, 0.48],[0.26, 0.55]] # first line [xs, ys]
    f1 = Polynomials.fit(l1p[1], l1p[2])
    l2p = [[0.41, 0.698],[0.225, 0.73]]
    f2 = Polynomials.fit(l2p[1], l2p[2])
    l3p = [[0.471, 0.865],[0.136, 0.82]]
    f3 = Polynomials.fit(l3p[1], l3p[2])
    display([f1(0.395), f2(0.617)])
    x = 0:0.01:1
    Plots.plot!(x, f1.(x), labels = "f1") # connects to p1 var
    annotate!(l1p[1][1], l1p[2][1], Plots.text("f1", 0.1, 10, :bottom, :right))
    Plots.plot!(x, f2.(x), labels = "f2")
    annotate!(l2p[1][1], l2p[2][1], Plots.text("f2", 0.1, 10, :bottom, :right))
    Plots.plot!(x, f3.(x), labels = "f3")
    annotate!(l3p[1][1], l3p[2][1], Plots.text("f3", 0.1, 10, :bottom, :right))
    # clustering of points to one of the lines:
    δy = 0.07
    abs_indices = trains ∪ tests 
    l1 = []; l2 = []; l3 = []
    for (i,id) ∈ enumerate(abs_indices)
        x = K[id, 1]; y = K[id, 2] 
        if abs(f1(x)-y) ≤ δy
            push!(l1, id)
        elseif abs(f2(x)-y) ≤ δy
            push!(l2, id)
        elseif abs(f3(x)-y) ≤ δy
            push!(l3, id)
        end
    end
    ls = [l1, l2, l3]
    # sort by pc2 (y axis):
    for (i,li) in enumerate(ls)
        sids = sortperm(K[li, 2])
        ls[i] = li[sids]
    end
    Plots.scatter!(K[ls[1],1], K[ls[1],2], markershape=:dtriangle, markercolor=:red, markersize = 5, labels = "class f1")
    Plots.scatter!(K[ls[2],1], K[ls[2],2], markershape=:utriangle, markercolor=:green, markersize = 5, labels = "class f2")
    Plots.scatter!(K[ls[3],1], K[ls[3],2], markershape=:ltriangle, markercolor=:yellow, markersize = 5, labels = "class f3")
    Plots.savefig(p1, "plot/deltaML/PCA_kernel_f1f2.png")
    display(p1)
    return ls # return the ids
end



# for convenicne:
function placemyimage(im, coor, scaling; centered=true)
    origin()
    Luxor.translate(coor) # flip y sign as usual
    @layer begin
        Luxor.scale(scaling)
        placeimage(im, Luxor.O, centered=centered)
    end
end

"""
rotate points THEN plot on 
"""
function main_rotate()
    # copy these to terminal, since 'dataset' is heavy to load
    K = readdlm("result/deltaML/PCA_kernel_2.txt") # n x 2 matrix of Real
    trains = Int.(vec(readdlm("data/tsopt/opt_tracker_freeze.txt")[2:end])) # m < n vector of Int
    # find angle α:
    p0 = [0.6290795, 0.01130475] # center of rotation
    p1 = [0., 0.5117841]
    p2 = [0., p0[2]]
    v1 = p1-p0
    v2 = p2-p0
    α = angle(v1, v2) # in radian
    display(α)
    # rotate all points of K:
    R = RotMatrix{2}(α)
    KR = (R*K')'
    # rotate a bit more to the right:
    p0 = [0.1833129, 0.4335298]; p1 = [0.09415505, 1.036599]; p2 = [p0[1], p1[2]]; 
    α = angle(p2-p0, p1-p0)
    R = RotMatrix{2}(-α)
    KR = (R*KR')'
    jp = jplot(jscatter(x=KR[trains,1], y=KR[trains,2], mode="markers"))
    display(jp)
    pp = Plots.scatter(KR[trains,1], KR[trains,2], xlabel="PC1", ylabel="PC2", markershape=:utriangle, markercolor=:red, legend=false, fmt = :svg)
    Plots.savefig(pp, "plot/deltaML/PCA_train.svg")
    #Plots.scatter(KR[trains,1], KR[trains,2])
    # auto bins by x with radius δ = 0.1:
    Kt = KR[trains,:]
    #δ = 0.1
    bounds = [floor(minimum(Kt[:,1])*10)/10, ceil(maximum(Kt[:,1])*10)/10] # first decimal roundings
    binranges = collect(range(bounds[1],bounds[2],step=0.1))
    bins = Dict() # relative index to Kt, can refer to idtrains[j ∈ bins]
    binx = Dict() 
    biny = Dict()
    for i ∈ eachindex(binranges)[1:end-1]
        bins[i] = []; binx[i] = []; biny[i] = []; 
    end
    for j ∈ axes(Kt, 1)
        x = Kt[j,1]
        for i ∈ eachindex(binranges)[1:end-1]
            if binranges[i] < x < binranges[i+1]
                push!(bins[i], j)#trains[j])
                push!(binx[i], Kt[j,1])
                push!(biny[i], Kt[j,2])
            end
        end
    end
    display(bins)
    display(biny)
    # sort each bin by y-axis:
    for (k,v) ∈ biny
        sid = sortperm(v)
        biny[k] = v[sid]
        binx[k] = binx[k][sid]
        bins[k] = bins[k][sid]
    end
    display(bins)
    display(biny)
    display(binx)    
    # [FOR LATER] find clusters within each bin: 

    # RYOIKI TENKAI: UNLIMITED DRAWING
    # try by simply rescale the x,y into pixel sizes and using the absolute coordinates
    boundx = [minimum(Kt[:,1]), maximum(Kt[:,1])]
    boundy = [minimum(Kt[:,2]), maximum(Kt[:,2])]
    display([boundx; boundy])
    f_rescale(z,mi,ma;sc=1.,shift=0.) = sc*(z-mi)/(ma-mi) + shift # fn to rescale to larger values
    Kt[:,1] = f_rescale.(Kt[:,1], boundx[1], boundx[2]; sc=2e3, shift=-1e3)
    Kt[:,2] = f_rescale.(Kt[:,2], boundy[1], boundy[2]; sc=2e3, shift=-1e3)
    display(Kt)
    display([minimum(Kt[:,1]), maximum(Kt[:,1])])

    # copypasta this to the terminal repeatedly until image looks good:
    # need to also check the delta for the molgraph "clusters" (see the paper for suggestions)
    # smiles MUST be loaded in the terminal: 
    dset = load("data/qm9_dataset.jld", "data"); smiless = map(d->d["smiles"], dset); tsmiless = smiless[trains]
    
    # ! draw for each bin:
    ## determine the uniform scaling for each bin, using c = (maxh - minh)/(sum(y) + (n-1)d), where the drawing LB need to be shifted to 0:
    svgs = readsvg.(drawsvg.(smilestomol.(tsmiless))) # preload all of the svg imagees
    hs = map(x -> x.height, svgs)
    dgap = 0.0 # distance gap in pixel unit
    miny = minimum(Kt[:,2])
    absminy = abs(miny) # for shifting the drawing by this magnitude so that the minimum is on 0 
    maxhs = [] # drawing upperbound that corresponds to each bin
    cs = [] # scaling coefficients for each bin !! can also experiment by manually editing the c in cs
    c_scale = 0.8 # 0 < s_c ≤ 1, the smaller then the image will be much closer
    for (i,kv) in enumerate(bins)
        id_maxh = bins[i][end] # get the id that corresponds to the image in the last index (the highest image)
        id_minh = bins[i][1] # minimum y of the bin
        maxh = Kt[id_maxh, 2] + hs[id_maxh]/2 # drawing upperbound of the bin
        minh = Kt[id_minh, 2] - hs[id_minh]/2 # drawing lowerbound of the bin
        c = (maxh - minh)/(c_scale*sum(hs[bins[i]]) + (length(bins[i])-1)*dgap)
        push!(maxhs, maxh); push!(cs, c)
        println([i, minh, maxh, c, length(bins[i])])
    end
    # coordinate placement of bins: (can be combined with the previous loop actually) 🤓
    pss = []
    c_overlap = 2.5 # c0 ≤ 2 means no overlap between images, c0 > 2 means there are overlaps
    for (i,kv) in enumerate(bins)
        bin = bins[i]; c = cs[i]
        n = length(bin)
        ps = zeros(n) # point locations
        ps[1] = Kt[bin[1],2] + c*(hs[bin[1]]/c_overlap)
        for i ∈ 2:n
            ps[i] = ps[i-1] + c*(hs[bin[i-1]]/c_overlap + hs[bin[i]]/c_overlap + dgap) 
        end
        push!(pss, ps)
    end
    # RYOIKI TENKAI: INFINITE DRAWING 📢 📢 🔥 🔥 🔥 🔥 🔥 🔥
    

    Drawing(2500, 2500, "pcagraph_scaled.svg")
    background("white")
    for (i,kv) in enumerate(bins)
        bin = bins[i]
        for (k,j) ∈ enumerate(bin) # for each j in bin
            placemyimage(svgs[j], Point(Kt[j,1], -pss[i][k]), cs[i])
        end
    end

    # add "axes lines" and texts manually
    sethue("black")
    origin()
    Luxor.arrow(Point(-1150, 1100), Point(1150, 1100); linewidth = 5, arrowheadlength = 20) # x axis
    Luxor.arrow(Point(-1100, 1150), Point(-1100, -1150); linewidth = 5, arrowheadlength = 20) # y axis
    fontsize(70)
    Luxor.text(("PC1 order"), Point(0,1200)) # x axis marker
    Luxor.text(("PC2 order"), Point(-1170,0), angle=-π/2) # y axis marker
    # superpose the original train PCA plot on top left: 
    ori_PCA = readsvg("plot/deltaML/PCA_train.svg")
    Luxor.translate(Point(-1075, -1150))
    @layer begin
        Luxor.scale(1.5)
        placeimage(ori_PCA, Luxor.O, centered=false)
    end
    finish()

    # CT MAXIMUM OUTPUT: ANIMATE! :
    info = (bins, svgs, Kt, pss, cs) # put important data here
    nfr = length(bins)
    anime = Movie(2500, 2500, "anime_pcagraph_scaled", 1:nfr)
    # tthis is the BACKGROUND of tthe animatoion
    function backdrop(scene, framenumber)
        background("white")
        sethue("black")
        origin()
        Luxor.arrow(Point(-1150, 1100), Point(1150, 1100); linewidth = 5, arrowheadlength = 20) # x axis
        Luxor.arrow(Point(-1100, 1150), Point(-1100, -1150); linewidth = 5, arrowheadlength = 20) # y axis
        fontsize(70)
        Luxor.text(("PC1 order"), Point(0,1200)) # x axis marker
        Luxor.text(("PC2 order"), Point(-1170,0), angle=-π/2) # y axis marker
        # superpose the original train PCA plot on top left: 
        ori_PCA = readsvg("plot/deltaML/PCA_train.svg")
        Luxor.translate(Point(-1075, -1150))
        @layer begin
            Luxor.scale(1.5)
            placeimage(ori_PCA, Luxor.O, centered=false)
        end
    end

    # here is the foreground anime
    # info contains the precomputed (images, coordinates, scalings)
    function mappa(scene, iframe, info)
        bins, svgs, Kt, pss, cs = info # unpack data
        # place images:
        for i ∈ 1:iframe
            bin = bins[i] # bins isa Dict(), i corresponds to bins' keys
            for (k,j) ∈ enumerate(bin) # bin isa vector, j corresponds to the row index in Kt
                placemyimage(svgs[j], Point(Kt[j,1], -pss[i][k]), cs[i])
            end
        end
    end

    Luxor.animate(anime,
            [
                Scene(anime,backdrop,1:nfr),
                Scene(anime,(sc,fr) -> mappa(sc,fr,info),1:nfr)
            ],
            creategif=true,
            framerate = 2,
            tempdirectory = "anime/molgraph",
            pathname = "anime/molgraph_JJK.gif"
        )

    # ! draw for each molecule indices (regardless of bins):
    #= Drawing(2500, 2500, "pcagraph.svg")
    background("white")
    for (i,train) in enumerate(trains)
        origin()
        Luxor.translate(Point(Kt[i,1],-Kt[i,2])) # flip y sign due to the coordinate system
        @layer begin
            Luxor.scale(1.0)
            placeimage(readsvg(drawsvg(smilestomol(tsmiless[i]))), Luxor.O, centered=true)
        end
    end
    finish() =#
end

function main_anime_ending()
    nfr = 45
    anime = Movie(1000, 650, "JJK_ending", 1:nfr)
    function backdrop_ed(scene, frame)
        background(0,0,0,0)
        #setopacity(1)
    end
    function anime_ed(scene, frame, nfr)
        Luxor.setline(5)
        Luxor.fontsize(55)
        Luxor.fontface("Noto Sans JP")
        # data:
        rates = [0.75, 0.5, 0.25]
        ys = [500, 625, 750, 875] .- 800
        sentences = ["Thank you", "Terima kasih", "ありがとうございます", "Danke schön"]
        
        # generate paths:
        pats = []
        for (i,s) ∈ enumerate(sentences)
            Luxor.textpath(sentences[i], Point(0, ys[i]), halign=:center, valign=:middle)
            pat = Luxor.storepath()
            Luxor.drawpath(pat, frame/nfr, action=:stroke)
            push!(pats, pat)
        end
        Luxor.fontsize(20)
        Luxor.text(("Special credits for slides:"), Point(-490, 240))
        # Luxor.jl:
        Luxor.fontsize(40)
        Luxor.textpath("Luxor.jl", Point(-120, 230), halign=:center, valign=:middle)
        sl = Luxor.storepath()
        Luxor.drawpath(sl, action=:stroke)
        Luxor.sethue("red")
        Luxor.setline(10)
        Luxor.setopacity(0.5)
        Luxor.drawpath(sl, frame/nfr,action=:stroke)
        # Reveal.js:
        Luxor.setline(5)
        Luxor.setopacity(1)
        Luxor.setcolor("black")
        Luxor.textpath("Reveal.js", Point(100, 230), halign=:center, valign=:middle)
        sl = Luxor.storepath()
        Luxor.drawpath(sl, action=:stroke)
        Luxor.sethue("green")
        Luxor.setline(10)
        Luxor.setopacity(0.5)
        Luxor.drawpath(sl, frame/nfr,action=:stroke)
    end
    Luxor.animate(anime,
            [
                Scene(anime,backdrop_ed,1:nfr),
                Scene(anime,(sc,fr) -> anime_ed(sc,fr,nfr),1:nfr)
            ],
            creategif=true,
            framerate = 60,
            tempdirectory = "anime/jjk_ed",
            pathname = "anime/JJK_ED.gif"
        )
end

"""
writes text like main_spiral_text
"""
function main_spiral_text()
    strs = readdlm("anime/refforanime.txt", '\n') # the strings
    strs = replace.(strs, "        \\item" => " ")
    Drawing(900, 900, "anime/naruto.svg")
    background("white")
    sethue("royalblue4") # hide
    fontsize(17)
    fontface("Menlo")
    textcurve(join(strs, "---"),
        -π,
        350, 450, 450,
        spiral_in_out_shift = -15.0,
        letter_spacing = 0,
        spiral_ring_step = -1)
    fontsize(35)
    fontface("Avenir-Black")
    textcentered("References", 450, 450)
    finish()
    preview()
end

"""
!! terminal
plot the delta Energy with anim??
"""
function main_plot_deltas()
    include("alouEt.jl")
    Random.seed!(603)
    E = vec(readdlm("data/energies.txt"))
    Fds_H_paths = ["atomref_features","featuresmat_bonds-H_qm9_post", "featuresmat_angles-H_qm9_post", "featuresmat_torsion-H_qm9_post"]
    Fs = map(Fd_path -> load("data/"*Fd_path*".jld", "data"), Fds_H_paths)
    idtrains = vec(readdlm("data/centers_30k_id57.txt", Int))[1:100]
    Eda = hp_baseline(E, Fs[1], Fs[2], Fs[3], Fs[4], idtrains; 
            sb = false, sn = false, st = false, 
            pb = false, pn = false, pt = false, 
            npb = 5, npn = 5, npt = 5)
    Edb = hp_baseline(E, Fs[1], Fs[2], Fs[3], Fs[4], idtrains; 
        sb = true, sn = false, st = false, 
        pb = false, pn = false, pt = false, 
        npb = 5, npn = 5, npt = 5)
    Edn = hp_baseline(E, Fs[1], Fs[2], Fs[3], Fs[4], idtrains; 
        sb = true, sn = true, st = false, 
        pb = false, pn = false, pt = false, 
        npb = 5, npn = 5, npt = 5)
    Edt = hp_baseline(E, Fs[1], Fs[2], Fs[3], Fs[4], idtrains; 
        sb = true, sn = true, st = true, 
        pb = false, pn = false, pt = false, 
        npb = 5, npn = 5, npt = 5)
    display([E Eda Edb Edn Edt])
    # sort by magnitude of E:
    yplots = [sort(abs.(E[idtrains])), sort(abs.(Eda[idtrains])), sort(abs.(Edb[idtrains])), sort(abs.(Edn[idtrains])), sort(abs.(Edt[idtrains]))]
    yplots = [log.(y) for y in yplots]
    p = Plots.plot(1:length(E[idtrains]), yplots,
                        ylims = [-10,10]
                        )
    #Plots.savefig(p, "plot/deltaML/deltaE.png")
    # try animate using Plots:
    gr()
    p = Plots.plot([cos for i ∈ eachindex(yplots)], 1, xlims = (0,100), ylims = (-10, 10),
                    markershape = [:xcross :cross :rect :auto :auto], markersize=4,
                    labels = permutedims([latexstring("E^{($(i-1))}") for i ∈ eachindex(yplots)]),
                    ylabel = latexstring("\\log(|E|)"),
                    dpi = 200)
    anim = Animation()
    for x = 1:100
        Plots.plot(push!(p, x, Float64[y[x] for y in yplots]))
        Plots.frame(anim)
    end
    gifpath = "anime/Edelta.gif"
    gif(anim, gifpath, fps=30)
    # paaste this in powershell:
    # ffmpeg -i "anime/Edelta.gif" -vsync 0 "anime/Edelta/%d.png" 
end


function main_plot_hpoptk()
    tb = readdlm("result/deltaML/tb_hpoptk_20240527T102359.txt")
    tb[17:end,1] .= "useq"
    xticks = tb[1:4, 4]; xtformat = string.(map(x -> @sprintf("%.0f",x), xticks))
    stids = 1:4:size(tb,1)
    ys = [[tb[i:i+3,12]] for i ∈ stids]; 
    yticks = vcat(12,[2.0^i for i ∈ 0:3]); ytformat = map(x -> @sprintf("%.0f",x), yticks)
    labels = permutedims([tb[i,1]*", "*tb[i,8] for i ∈ stids])
    p = Plots.plot(xticks, ys,
            yticks = (yticks, ytformat),
            xticks = (xticks, xtformat), 
            xaxis = :log, yaxis = :log,
            linestyle = [:dash :dash :dash :dash :solid :solid :solid :solid],
            markershape = [:circle :rect :utriangle :diamond :circle :rect :utriangle :diamond], 
            markercolor = [:black :green :blue :purple :black :green :blue :purple],
            linecolor = [:black :green :blue :purple :black :green :blue :purple],
            markersize = (ones(8)*6)',
            labels = labels, xlabel = "Ntrain", ylabel = "MAE (kcal/mol)",
            title = "Random vs Usequence", legend = :outertopright,
            dpi = 1000
        )
    display(p)
    Plots.savefig(p, "plot/deltaML/fig_RandVFPS_1k.png")
end


"""
main function to plot the 10x10 molgraphs sorted by \theta,
and the PCA using molgraph
!! as usual, preferrable to be pasted to terminal
"""
function main_plot_molgraph()
    dset = load("data/qm9_dataset.jld", "data")
    idtrains = Int.(vec(readdlm("data/tsopt/opt_tracker_freeze.txt")[2:end])) # optimized training set
    smiless = map(d->d["smiles"], dset) # selected smiless from t
    θ = vec(readdlm("data/theta_best.txt")) # best weights obtained
    sid = reverse(sortperm(θ)) # descending sort
    ss = smiless[idtrains][sid]

    # compute atomization energies:
    include("alouEt.jl")
    E = readdlm("data/energies.txt")
    F_dresseds = [load("data/atomref_features.jld", "data"), [], [], []]
    ΔE, Et = hp_baseline(E, F_dresseds[1], F_dresseds[2], F_dresseds[3], F_dresseds[4], idtrains; get_eatom=true) # in kcal/mol
    Et *= 627.509
    str_Et = clean_float(Et)
    str_θ = format_string_float.(3, θ) # try 3 decimals
    # Ryoiki Tenkai: DRAW
    gridsize = (10, 10) # num of grids (row, col)
    ptsize = (130, 180) # size of partition/cell (w, h)
    imgsize = (ptsize[1]*gridsize[1], ptsize[2]*gridsize[2]) # total size of image (csize, rsize)
    Drawing(imgsize[1], imgsize[2], "molgraphs2.svg")
    background("white")
    origin()
    t = Table(gridsize, ptsize)
    fontsize(15)
    fontface("Times-Roman")
    for (pt, i) in t
        img = readsvg(drawsvg(smilestomol(ss[i])))
        placemyimage(img, pt - (0., 30.), .7; centered=true)
        origin()        
        Luxor.text("#"*string(idtrains[sid][i]), pt + (0., 28.) , halign=:center, valign=:middle) # mol label
        Luxor.text(latexstring("E = $(str_Et[idtrains][sid][i])"), pt + (0., 46.) , halign=:center, valign=:middle) # atomization energy
        Luxor.text(latexstring("w = $(str_θ[sid][i])"), pt + (0., 64.) , halign=:center, valign=:middle) # atomization energy
        println([idtrains[sid][i], ss[i], θ[sid][i], i]) #$(str_Et[idtrains][sid][i])
    end
    finish()
end

"""
plot the histogram of atomization 100 mol vs QM9 dataset
"""
function main_plot_histograms()
    E = readdlm("data/energies.txt")
    idtrains = Int.(vec(readdlm("data/tsopt/opt_tracker_freeze.txt")[2:end]))
    F_dresseds = [load("data/atomref_features.jld", "data"), [], [], []]
    ΔE, Et = hp_baseline(E, F_dresseds[1], F_dresseds[2], F_dresseds[3], F_dresseds[4], idtrains) # in kcal/mol
    Et *= 627.509
    b_range = range(minimum(Et) - 100., maximum(Et) - 100., length=51) # minimum and maximum from inspecting the data manually
    h = Plots.histogram(Et, label="130k QM9 molecules", bins=b_range, normalize=:probability, color=:green, xlabel=latexstring("E^{(DA)}"), ylabel=L"$P (E^{(DA)} )$", la=0.5, lw=0.5, dpi=1000)
    Plots.stephist!(Et[idtrains], label="100 selected molecules", bins=b_range, normalize=:probability, color=:red, lw=2)
    Plots.savefig(h, "plot/deltaML/hist_Eatom.svg")
end

"""
copy paste the content of this function to the terminal to run
"""
function terminal_get_pattern()
    dataset = load("data/qm9_dataset.jld", "data")

    # get pattern of each diagonal:
    l1,l2 = main_PCA_plot()
    l1forms = map(d->d["formula"], dataset[l1])
    l2forms = map(d->d["formula"], dataset[l2])
    # l3 here
    tb = Matrix{String}(undef, length(l2), 2)
    tb[:,2] .= l2forms
    tb[1:length(l1),1] = l1forms
    tb[length(l1)+1:end,1] .= ""
    writedlm("result/deltaML/PCA_exp.txt", tb)

    # get pattern between diagonals:
    ls = [l1,l2,l3]
    ns = []
    for l in ls
        push!(ns, map(d->d["n_atom"], dataset[l]))
    end
    cs = counter.(ns)
    unioncs = sort(collect(reduce(∪ , keys.(cs) )))
    cmat = zeros(Int, length(unioncs), length(cs)+1)
    cmat[:,1] = unioncs
    for (i,el) in enumerate(unioncs)
        for (j,c) in enumerate(cs)
            cmat[i,j+1] = c[el]
        end
    end
    display(cmat)
    writelatextable(cmat, "result/deltaML/PCA_mol_freq.tex"; hline=false)
end

function main_get_timing_table()
    tb = readdlm("result/deltaML/MAE_enum_v2_combined_101123.txt")
    # query kernel timing for each feature:
    fs = unique(tb[2:end,8])
    display(fs)
    ts_g = [] # mean timing of each kernel
    ts_d = []
    for f in fs
        push!(ts_g,mean(tb[query_indices(tb, [8,4,7],[f, 30_000,"GK"]),13]))
        push!(ts_d,mean(tb[query_indices(tb, [8,4,7],[f, 30_000,"DPK"]),13]))
    end
    display([fs ts_g ts_d])
    out = [fs ts_g ts_d]
    out[:,[2,3]] .= clean_float.(out[:,[2,3]])
    display(out)
    writelatextable(out, "result/deltaML/tb_timing_30k100k.tex", hline = false)
end


"""
plot the fobj of hpopt, 3 plots (with ACSF, MBDF, CMBDF), see the hpopt_* folders
"""
function main_plot_fs()
    tbs = ["data/hpopt_111023/sim/sim_tracker.txt", "data/hpopt_161023_5kcalmol/sim/sim_tracker.txt", "data/hpopt_081123_3.67kcalmol/sim/sim_tracker.txt"]
    fts = ["ACSF", "MBDF", "CMBDF"]
    for (i,tb) ∈ enumerate(tbs)
        # write (init,best) table:
        tb = readdlm(tb)
        initpoint = tb[1,:]
        yminid = argmin(tb[:,3])
        minpoint = tb[yminid, :]
        display(hcat(initpoint, minpoint)')
        display([yminid, size(tb,1)])
        ymin = tb[yminid, 3]
        # plot f(x)
        y = tb[:,3]
        p = plot(eachindex(y), y, 
            ylimits=(0,20), xlimits=(yminid-100,yminid+100), 
            xlabel = "iter", ylabel = "f(x)", label = false, dpi=1000)
        scatter!([yminid], [ymin], markercolor = :red, markersize = 5, labels = clean_float(ymin)*" kcal/mol", legend=:bottomright)
        display(p)
        savefig(p, "plot/deltaML/hpopt_"*fts[i]*".png")
    end
end



"""
=====================
ROSEMI data analysis:
=====================
"""

"""
compare rosemi against ratpots and chipr

data pre 07/11/2024: rnew = vcat(load("result/hdrsm_20240502T122032.jld", "data"), load("result/hdrsm_singlet_H2_2.jld", "data"))
post: 
"""
function main_tb_hxoy_rerun()
    dset = load("data/smallmol/hxoy_data_req.jld", "data")
    rold = load("result/hxoy_5fold_result.jld", "data") # dict of list
    rnew = vcat(load("result/hdrsm_20241107T124925.jld", "data"), load("result/hdrsm_singlet_H2_2_0711.jld", "data")) #load("result/hxoy_diatomic_rosemi_rerun.jld", "data") # list of dict

    # output tb (no header, add in the end):
    tb = Matrix{Any}(undef, length(rnew), 2*4) # row entries = [[ min, median, mean, max] of RMSE of [ROSEMI, CHIPR, and maybe RATOPTS] ]
    
    # statistics of rosemi:
    mins = []; medians = []; maxs = []; means = []
    for r ∈ rnew
        push!(mins, minimum(r["RMSE"])); push!(medians, median(r["RMSE"])); push!(maxs, maximum(r["RMSE"])); push!(means, mean(r["RMSE"]));
    end
    tb[:,1] = mins; tb[:,2] = medians; tb[:,3] = means; tb[:,4] = maxs
    #display(tb)
    # stats of ratpots and chipr:
    qkeys = ["chipr_acc"] #["ansatz_1_acc", "ansatz_2_acc", "chipr_acc"]
    # find matching indices:
    mols = map(d->d["mol"], rnew)
    ids = map(mol -> findall(mol .== rold["mol"])[1], mols)
    #display(ids)
    #display(rold["mol"][ids])
    slices = [5*(i-1)+1:5*(i-1)+5 for i ∈ ids]
    #display(slices)
    # bin each 5 indices to one set:
    itb = 5
    for k ∈ qkeys
        mins = []; medians = []; maxs = []; means = []
        for s ∈ slices
            data = rold[k][s]
            push!(mins, minimum(data)); push!(medians, median(data)); push!(maxs, maximum(data)); push!(means, mean(data));
        end
        tb[:,itb] = mins; tb[:, itb+1] = medians; tb[:, itb+2] = means; tb[:, itb+3] = maxs;
        itb += 4
        #display(tb)
        println(k)
        println(means)
    end
    #display(tb)
    Base.permutecols!!(tb, [1,5,2,6,3,7,4,8]) # swap posiitons to group the columns' category
    #display(tb)
    # find the winner between ROSEMI and CHIPR for each row:
    ws = [] # for each row 4 entries
    for i ∈ axes(tb, 1)
        w = []
        for j ∈ 1:2:8 # 4 measurements
            comp = [j,j+1]
            imi = argmin(tb[i,comp])
            push!(w, comp[imi])
        end
        push!(ws, w)
    end
    #display(ws)
    # write to latextable: !!! DISABLE (comment out) when generating plot:
    display(tb)
    tb = format_string_float.(1,tb; scientific=true)
    for i ∈ axes(tb, 1)
        tb[i,ws[i]] .= latex_bold.(tb[i,ws[i]])
    end
    # get the indices of each molecule relative to the dataset:
    dset_ids = []
    for mol ∈ mols
        for j ∈ eachindex(dset)
            if mol == dset[j]["mol"]
                push!(dset_ids, j)
                break
            end
        end
    end
    # edit molecule names:
    #display([mols dset_ids])
    molperm = reduce(vcat, permutedims.(split.(mols, "_"))) # split the molecule id
    molstr, molid = (molperm[:,1], molperm[:,2])
    ids = sortperm(molstr) # sort by molname
    molstr = latex_chemformat.(molstr) # format chem
    # sort and join the strings back:
    molstr = molstr[ids]
    molid = molid[ids]
    mol_dsetids = dset_ids[ids]
    molstates = map(x->x["state"], dset[mol_dsetids])
    mol_ndatas = map(x->length(x["V"]), dset[mol_dsetids])
    molstr = map((l,m,n) -> "("*string(l)*") "*m*" "*n, eachindex(molstr), molstr, molstates) # join mol string with its state 
    #display([molstr mol_ndatas])
    #mols = map((x,y)-> x*raw"$^{"*y*raw"}$", molstr, molid)
    #display(mols)
    # permute rows of table:
    tb = permutedims(tb)
    Base.permutecols!!(tb, ids)
    tb = permutedims(tb)
    # join molname, ndata, and table:
    tb = hcat(mol_ndatas, tb)
    tb = hcat(molstr, tb)
    display(tb)
    writelatextable(tb, "result/tb1_hxoy_rerun_0711.tex"; hline=false)
    ##########################
    # Ratio plot:
    #= inds = collect(3:2:9) # correspond to each measurement category
    ratios = zeros(axes(tb, 1), length(inds)) # each row is one dataset, each column is one measurement category
    display([tb[:,inds[1]] tb[:, inds[1]+1] tb[:,inds[1]] ./ tb[:, inds[1]+1]])
    for i in axes(ratios, 2)
        ratios[:,i] .= tb[:,inds[i]] ./ tb[:,inds[i]+1]
    end
    display(ratios)
    p = Plots.plot(axes(ratios, 1), [ratios[:,i] for i in axes(ratios, 2)], 
                yaxis=:log10, 
                yticks=([minimum(ratios), 1e-3, 0.1, 1, 10, maximum(ratios)], [5.4e-6, 1e-3, 0.1, 1, 10, round(maximum(ratios), digits=1)]),
                xticks=(axes(ratios, 1), molstr), xrot=40,
                ylabel = "Ratio of ROSEMI/CHIPR RMSEs",
                labels = ["min" "median" "mean" "max"],
                legend = :bottomright,
                linestyle = [:dash :dash :solid :solid],
                markershape = [:circle :rect :utriangle :diamond], 
                markercolor = [:black :green :blue :purple],
                linecolor = [:black :green :blue :purple],
                dpi=1000, xtickfontsize=7
            )
    Plots.savefig(p, "plot/ROSEMIvCHIPR.svg") =#
end

"""
display hyperparameter optimization table of pair hxoy data (pretty much similar with the above function)
usage of table exaample: tb = readdlm("data/smallmol/hpopt_rsm_hxoy_20240501T185604.text", '\t')
"""
function main_tb_hxoy_hpopt()
    dset = load("data/smallmol/hxoy_data_req.jld", "data")
    tb = readdlm("data/smallmol/hpopt_rsm_hxoy_20241107T124925.text", '\t')
    mols = tb[:,1]
    # get the indices of each molecule relative to the dataset:
    dset_ids = []
    for mol ∈ mols
        for j ∈ eachindex(dset)
            if mol == dset[j]["mol"]
                push!(dset_ids, j)
                break
            end
        end
    end
    molperm = reduce(vcat, permutedims.(split.(mols, "_"))) # split the molecule id
    molstr, molid = (molperm[:,1], molperm[:,2])
    ids = sortperm(molstr) # sort by molname
    molstr = latex_chemformat.(molstr) # format chem
    # sort and join the strings back:
    molstr = molstr[ids]
    molid = molid[ids]
    mol_dsetids = dset_ids[ids]
    molstates = map(x->x["state"], dset[mol_dsetids])
    molstr = map((l,m,n) -> "("*string(l)*") "*m*" "*n, vcat(1, 3:length(molstr)+1), molstr, molstates) # exclude the 2nd H2 (was not hyperopt'd)
    tb = tb[:,3:end]
    tb[1:end, 1] .= format_string_float.(1,tb[1:end, 1]; scientific=true)
    tb[1:end, 2] .= format_string_float.(1,tb[1:end, 2]; scientific=false)
    tb[1:end, end] .= format_string_float.(1,tb[1:end, end]; scientific=false)
    tb = permutedims(tb)
    Base.permutecols!!(tb, ids)
    tb = permutedims(tb)
    tb = hcat(molstr, tb)
    display(tb)
    writelatextable(tb, "result/tb_hxoy_hpopt_0711.tex"; hline=false)
end

function main_rosemi_hn()
    rold = load("result/hn_results_old.jld", "data")
    rnew = load("result/hn_rosemi_rerun.jld", "data")

    # output tb (no header, add in the end):
    tb = Matrix{Any}(undef, length(rnew), 6) # row entries = [[ min, median, max] of RMSE of [ROSEMI, RATPOT1, RATPOT2, CHIPR] ]
    
    # statistics of rosemi:
    mins = []; medians = []; maxs = []
    for r ∈ rnew
        push!(mins, minimum(r["RMSE"])); push!(medians, median(r["RMSE"])); push!(maxs, maximum(r["RMSE"]))
    end
    tb[:,2] = mins; tb[:,3] = medians; tb[:,4] = maxs
    display(tb)
    mols = map(r->r["mol"], rnew)
    mols = latex_(mols)
    tb[:,1] = mols
    tb[:, [end-1,end]] .= "N/A" # insert the rest manually
    tb[:, [2,3,4]] .= convert_to_scientific(tb[:, [2,3,4]])
    display(tb)
    writelatextable(tb, "result/tb_hn_rerun.tex")
end

"""
figure out the equilibrium distances for each dataset
"""
function main_eq_dist()
    data = load("data/smallmol/hxoy_data.jld", "data")
    # check using the lowest energy query:
    mols = map(d-> d["mol"], data)
    umols = unique(mols)
    for smol in umols
        reqs = []
        for (i,d) in enumerate(data)
            if d["mol"] == smol
                veqid = argmin(d["V"])
                req = d["R"][veqid]
                push!(reqs, req)
                println([req, d["mol"],i])
            end
        end
    end
    # include some indices:
    iids = [1,2,3,11,12,13,14,15,16,17,20,21,23,24,25]
    data = data[iids]
    mols = map(d-> d["mol"], data)
    umols = unique(mols)
    for smol in umols
        reqs = []
        for (i,d) in enumerate(data)
            if d["mol"] == smol
                veqid = argmin(d["V"])
                req = d["R"][veqid]
                d["req"] = req
                push!(reqs, req)
                println([req, d["mol"],i])
            end
        end
    end
    display(map(d->d["req"], data))
    # double check using visual:
    for d in data
        if d["mol"] == "O2"
            d["R"] = d["R"][1:end-1]; d["V"] = d["V"][1:end-1] # remove end outliers
            d["note"] = "removed (R,V)[end], outlier at faraway distance"
            p = jplot(jscatter(x=d["R"][1:end-1], y=d["V"][1:end-1], mode="lines"))
            display(d["mol"])
            display(p)
            display(d)
        end
    end
    # save reselected data:
    save("data/smallmol/hxoy_data_req.jld", "data", data) 
end