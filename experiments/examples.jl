using Base: upperbound
using MixedPrecisionPCG
using LinearAlgebra, Random, BFloat16s, DataFrames, JSONTables

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
errorbound1         = upper_bound(ex1, ErrorBound)
residualbound1      = upper_bound(ex1, ResidualBound)

# Experiment 2: Split PCG comparison.
splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s)]
ex2                 = Experiment(85, 1.0, 1e+5, 0.6, 65, 350, splitpcg_precisions)
ads2                = runpcgexperiments(ex2, Split)
ads2saad            = runpcgexperiments(ex2, SaadSplit)
errorbound2         = upper_bound(ex2, ErrorBound)
residualbound2      = upper_bound(ex2, ResidualBound)


#M        = Symmetric(L * L')
