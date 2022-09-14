
using DelimitedFiles

"""
usual main caller
"""
function main()
    # read gdb file:
    f = open("data/qm9/dsgdb9nsd_000002.xyz")
    lines = readlines(f)
    display(lines)
    #006140 000001
    fd = readdlm("data/qm9/dsgdb9nsd_000001.xyz", '\t', String, '\n')
    display(fd[2, 1:16])
end

main()