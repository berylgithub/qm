using Pkg, DelimitedFiles

deps = vec(readdlm("dependencies.txt"))
display(deps)
for dep ∈ deps
    Pkg.add(dep)
end