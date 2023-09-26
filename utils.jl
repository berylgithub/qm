"""
all kinds of utility functions
"""

using LaTeXStrings, Printf, DelimitedFiles, JLD

# get variable name
macro Name(arg)
    string(arg)
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
function writelatextable(table, filename; hline = true)
    open(filename, "w") do io
        for i ∈ axes(table, 1)
            str = ""
            for j ∈ axes(table, 2)
                str *= string(table[i, j])*"\t"*"& "
            end
            str = str[1:end-2]
            str *= raw" \\ "
            if hline
                str *= raw" \hline"
            end
            str *= "\n"
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

# cleans the structure data of floats into 3 digits-behind-comma format
function clean_float(data)
    return map(data) do el
        s = ""
        if occursin("e", string(el)) # retain scinetfigc notaioton
            s = @sprintf "%.3e" el
        else
            s = @sprintf "%.3f" el
        end
        s
    end
end

# add \ for latex underscore escape, only handles findfirst for now
function latex_(data)
    return map(data) do el
        id = findfirst("_", el)
        if id !== nothing
            s = el[1 : id[1]-1]*raw"\_"*el[id[1]+1 : end]
        else
            s = el
        end
        s
    end
end

"""
=== query functions ===
"""

"""
very specific function, may be changed at will
query the information from a table of the row with the minimum MAE
"""
function query_min_f(table; feature_type = "")
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
more generic query min function, returns the minimum fobj given the selected columns 
params:
    - table: matrix containing info
    - colids: list of column ids selected for query
    - colnames: list of name of the columns in which we want to look the minimum from
"""
function query_min(table, colids, colnames)
    selids = query_indices(table, colids, colnames)
    minid = argmin(table[selids, 7])
    return selids[minid] # the nth index of the selected indices
end

function main_convert_datatype()
    fpaths = ["ACSF_51", "SOAP", "FCHL19"]
    for fp ∈ fpaths
        f = load("data/"*fp*".jld", "data")
        f = Matrix{Float64}.(f)
        save("data/"*fp*".jld", "data", f)
    end
end