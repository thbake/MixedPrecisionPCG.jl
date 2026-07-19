using Base: upperbound
using MixedPrecisionPCG
using BFloat16s, LinearAlgebra, Random

Random.seed!(12349301)

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
ex1                 = Experiment{Left}(85, 1.0, 1e5, 0.6, 55, 2500, leftpcg_precisions)
ads1                = runpcgexperiments(ex1)

write_to_file("convergence1.json", ads1, ex1)

# Experiment 2: Split PCG comparison.
splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s)]
ex2                 = Experiment{Split}(85, 1.0, 1e5, 0.6, 65, 350, splitpcg_precisions)
ex2saad             = Experiment{SaadSplit}(ex2)
ads2                = runpcgexperiments(ex2)
ads2saad            = runpcgexperiments(ex2saad)

write_to_file("convergence2.json",     ads2,     ex2)
write_to_file("convergence2saad.json", ads2saad, ex2)

# Experiment 3
splitpcg_precisions = [(d,d), (s,s), (h_b16, h_b16), (d,s), (d, h_b16), (s, h_b16)]
ex3  = Experiment{Split}(85, 1.0, 1e5, 0.6, 55, 2500, splitpcg_precisions)
ads3 = runpcgexperiments(ex3)

write_to_file("convergence3.json", ads3, ex3)

# Experiment 4: Split PCG for all possible configurations
splitpcg_precisions = [ (prec1, prec2) for prec1 in leftpcg_precisions for prec2 in leftpcg_precisions ]
ex4                 = Experiment{Split}(85, 1.0, 1e5, 0.6, 55, 2500, splitpcg_precisions)
ads4                = runpcgexperiments(ex4)

# Transform to forward and backward errors for heatmaps
write_heatmap_data("heatmaps_data.json", ex4, ads4)


# Adapted experiment of Epperly, Greenbaum and Nakatsukasa.
n       = 100; # Reduce problem size due to extreme slow unpreconditioned CG convergence.
cond_A  = 1e12;
sigma   = diagm( 10.0 .^(range(0, -log10(cond_A), n)) );

# Create Haar orthogonal matrix.
QU, RU  = qr( randn(n,n) );
U       = QU * diagm( sign.(diag(RU)) );

A       = Hermitian(U * sigma * U');

# Construct preconditioner.
G       = randn(4n, n);
tmp     = diagm( 10.0 .^(range(0, -log10(cond_A) / 2, length=n)) );
sigmaP  = tmp * (G' * G) * tmp;
P       = Hermitian(U * sigmaP * U');
R       = cholesky(P);
x       = randn(n);
x0      = zeros(n);
b       = A * x;
maxiter = 50;

ls  = LinearSystem(A, b, x, x0);
ex  = Experiment{Left}(ls, R.L, maxiter, [d, s]);
ads = runpcgexperiments(ex)

write_to_file("convergence.json", ads, ex)

# Run unpreconditioned instance.
ex.max_iter       = 200_000;
ex.preconditioner = I(n);
ex.precisions     = [d]
ads_unprec        = runpcgexperiments(ex)

write_to_file("unprecond_convergence.json", ads_unprec, ex)

