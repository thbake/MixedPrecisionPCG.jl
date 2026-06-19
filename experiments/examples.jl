using Base: upperbound
using MixedPrecisionPCG
using BFloat16s, LinearAlgebra, Random

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
#leftpcg_precisions  = [d, s, h, h_b16]
leftpcg_precisions  = [h]
ex1                 = Experiment{Left}(85, 1.0, 1e+5, 0.6, 55, 2, leftpcg_precisions)
ads1                = runpcgexperiments(ex1)


write_to_file("convergence1.json", ads1, ex1)

## Experiment 2: Split PCG comparison.
#splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s)]
#ex2                 = Experiment{Split}(85, 1.0, 1e+5, 0.6, 65, 350, splitpcg_precisions)
#ex2saad             = Experiment{SaadSplit}(ex2)
#ads2                = runpcgexperiments(ex2)
#ads2saad            = runpcgexperiments(ex2saad)
#
#write_to_file("convergence2.json",     ads2,     ex2)
#write_to_file("convergence2saad.json", ads2saad, ex2)
#
#
## Experiment 3: Split PCG for all possible configurations
#splitpcg_precisions = [ (prec1, prec2) for prec1 in leftpcg_precisions for prec2 in leftpcg_precisions ]
#ex3                 = Experiment{Split}(85, 1.0, 1e+5, 0.6, 55, 2500, splitpcg_precisions)
#ads3                = runpcgexperiments(ex3)
#
#write_to_file("convergence3.json", ads3, ex3)
#
## Transform to forward and backward errors for heatmaps
#write_heatmap_data("heatmaps_data.json", ex3, ads3)
#
## Experiment 4: Slightly ill-conditioned matrices
#ex4                 = Experiment{Left}(85, 1.0, 1e+10, 0.6, 70, 2500, leftpcg_precisions)
#perturbation        = 1e-3 .* rand( size(ex4.ls.A)... )
#perturb_preconditioner!(ex4, perturbation)
#ads4                = runpcgexperiments(ex4)
#
#
#write_to_file("convergence4.json", ads4, ex4)
#
#ex5                 = Experiment{Split}(85, 1.0, 1e+10, 0.6, 25, 8000, [(d, d), (d, s), (s, d), (s, s)])
#ads5                = runpcgexperiments(ex5)
#
#write_to_file("convergence5.json", ads5, ex5)
