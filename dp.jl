using JLD

"""
usual main caller
"""
function main()
    # read gdb file:
    f = open("data/qm9/dsgdb9nsd_000002.xyz")
    lines = readlines(f)
    display(lines)
end

main()