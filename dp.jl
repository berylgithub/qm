using PyCall, ASE, ACSF

"""
usual main caller
"""
function main()
    # read gdb file:
    #f = open("data/qm9/dsgdb9nsd_000002.xyz")
    #lines = readlines(f)
    #display(lines)
    # test load pickle from python
    math = pyimport("math")
    println(math.sin(math.pi / 4)) # returns ≈ 1/√2 = 0.70710678...

    py"""
    import numpy as np
    
    def sinpi(x):
        return np.sin(np.pi * x)
    """
    py"sinpi"(1)
end

main()