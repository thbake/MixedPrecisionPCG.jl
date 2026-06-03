using Base: upperbound
using MixedPrecisionPCG
using LinearAlgebra, Random, BFloat16s

Random.seed!(1234)

# Example: Strakos Matrix with eigenvalues accumulated to the left.
# ==================================================================

# Set precisions
d     = Float64
s     = Float32
h     = Float16
h_b16 = BFloat16

# Set unit roundoff values
ud   = 0.5 * eps(d)
us   = 0.5 * eps(s)
uh   = 0.5 * eps(h)
ub16 = 0.5 * eps(h_b16)


# Experiment 1: Left PCG curves.
leftpcg_precisions  = [d, s, h, h_b16]
ex1                 = Experiment(85, 1.0, 1e+5, 0.6, 55, 2500, leftpcg_precisions)
ads1                = runpcgexperiments(ex1, Left)
errorbound1         = upper_bound(ex1, Error)
residualbound1      = upper_bound(ex1, Residual)

write_to_file("errornorm1.json",   ads1,    errorbound1, ex1, Left,    Error)
write_to_file("trueresnorm1.json", ads1, residualbound1, ex1, Left, Residual)

# Experiment 2: Split PCG comparison.
splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s)]
ex2                 = Experiment(85, 1.0, 1e+5, 0.6, 65, 350, splitpcg_precisions)
ads2                = runpcgexperiments(ex2, Split)
ads2saad            = runpcgexperiments(ex2, SaadSplit)
errorbound2         = upper_bound(ex2, Error)
residualbound2      = upper_bound(ex2, Residual)

write_to_file("errornorm2.json",           ads2,    errorbound2, ex2,     Split,    Error)
write_to_file("trueresnorm2.json",         ads2, residualbound2, ex2,     Split, Residual)

write_to_file("errornorm2saad.json",   ads2saad,    errorbound2, ex2, SaadSplit,    Error)
write_to_file("trueresnorm2saad.json", ads2saad, residualbound2, ex2, SaadSplit, Residual)


