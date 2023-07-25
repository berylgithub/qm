"""
all kinds of utility functions
"""

using LaTeXStrings, Printf, DelimitedFiles

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