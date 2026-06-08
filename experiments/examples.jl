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


## Experiment 1: Left PCG curves.
leftpcg_precisions  = [d, s, h, h_b16]
ex1                 = Experiment{Left}(85, 1.0, 1e+5, 0.6, 55, 2500, leftpcg_precisions)
ads1                = runpcgexperiments(ex1)
errorbound1         = upper_bound(ex1, Error)
residualbound1      = upper_bound(ex1, Residual)

write_to_file("errornorm1.json",   ads1,    errorbound1, ex1,    Error)
write_to_file("trueresnorm1.json", ads1, residualbound1, ex1, Residual)

# Experiment 2: Split PCG comparison.
splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s)]
ex2                 = Experiment{Split}(    85, 1.0, 1e+5, 0.6, 65, 350, splitpcg_precisions)
ex2saad             = Experiment{SaadSplit}(85, 1.0, 1e+5, 0.6, 65, 350, splitpcg_precisions)
ads2                = runpcgexperiments(ex2)
ads2saad            = runpcgexperiments(ex2saad)
errorbound2         = upper_bound(ex2, Error)
residualbound2      = upper_bound(ex2, Residual)

write_to_file("errornorm2.json",           ads2,    errorbound2, ex2,    Error)
write_to_file("trueresnorm2.json",         ads2, residualbound2, ex2, Residual)

write_to_file("errornorm2saad.json",   ads2saad,    errorbound2, ex2saad,    Error)
write_to_file("trueresnorm2saad.json", ads2saad, residualbound2, ex2saad, Residual)

# Experiment 3: Split PCG for all possible configurations
splitpcg_precisions = [ (prec1, prec2) for prec1 in leftpcg_precisions for prec2 in leftpcg_precisions ]
ex3                 = Experiment{Split}(85, 1.0, 1e+5, 0.6, 55, 2500, splitpcg_precisions)
ads3                = runpcgexperiments(ex3)
errorbound3         = upper_bound(ex3, Error)
residualbound3      = upper_bound(ex3, Residual)

write_to_file("errornorm3.json",           ads3,    errorbound3, ex3,    Error)
write_to_file("trueresnorm3.json",         ads3, residualbound3, ex3, Residual)

# Transform to forward and backward errors for heatmaps
write_heatmap_data("heatmaps_data.json", ex3, ads3)

# Experiment 4: Slightly ill-conditioned matrices
ex4                 = Experiment{Left}(85, 1.0, 1e+10, 0.6, 70, 200, leftpcg_precisions)
ads4                = runpcgexperiments(ex4)
errorbound4         = upper_bound(ex4,    Error)
residualbound4      = upper_bound(ex4, Residual)


write_to_file("errornorm4.json",   ads4,    errorbound4, ex4,    Error)
write_to_file("trueresnorm4.json", ads4, residualbound4, ex4, Residual)

ex5                 = Experiment{Split}(85, 1.0, 1e+10, 0.6, 25, 8000, [(d, d), (d, s), (s, d), (s, s)])
ads5                = runpcgexperiments(ex5)
errorbound5         = upper_bound(ex4,    Error)
residualbound5      = upper_bound(ex4, Residual)

write_to_file("errornorm5.json",   ads5,    errorbound5, ex5,    Error)
write_to_file("trueresnorm5.json", ads5, residualbound5, ex5, Residual)




