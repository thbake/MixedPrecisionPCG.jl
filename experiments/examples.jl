using MixedPrecisionPCG
using LinearAlgebra, MATLAB

# System 1
# =============================================================================

n1    = 40
kappa = 10^5

# Construct matrix with prescribed eigenvalues.
A1 = geomdist_eigvalmatrix(kappa, n1)

# Choose random solution.
x1 = rand(n1)

# Get right-hand side.
b1 = A1 * x1

# Set initial guess to zero
x0_1 = zeros(n1)

# Maximal number of iterations
max_iter1 = 150

# Assemble linear algebraic system
ls1 = LinearSystem(A1, b1, x1, x0_1)

# Create low precision Cholesky factorization as preconditioner
L1 = low_precision_preconditioner(A1, tol = 1e-4)

# System 2
# =============================================================================

# Construct SPD, banded coefficient matrix.
A2 = mat"delsq(numgrid('S', 102))"

n2 = size(A2, 1)

b2 = ones(n2)

x2 = A2 \ b2

x0_2 = zeros(n2)

max_iter2 = 150

ls2 = LinearSystem(A2, b2, x2, x0_2)

# Construct preconditioner using MATLAB routine ichol (incomplete Cholesky).
L2  = mat"ichol($A2)"

# System 3: System 2 with a different preconditioner that accelerates convergence.
# =============================================================================

# Generate preconditioner using a modified incomplete Cholesky factorization.
L3 = mat"ichol($A2, struct('michol', 'on'))"

# Set precisions
d = Float64
s = Float32
h = Float16

splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s), (d, h)]
leftpcg_precisions  = [d, s, h]

#v_ls       = [      ls1,       ls2,       ls2]
#v_prec     = [       L1,        L2,        L3]
#iterations = [max_iter1, max_iter2, max_iter2]
v_ls       = [ls1, ]
v_prec     = [L1, ]
iterations = [max_iter1, ]

v_ad_left  = runpcgexperiments(Left,  v_ls, v_prec, iterations, leftpcg_precisions)
v_ad_split = runpcgexperiments(Split, v_ls, v_prec, iterations, splitpcg_precisions)

#splitpcg_precisions = [(s, d)]
kappa_range         = 10.0 .^collect(2:2:10)

#ad_vec, kappa_range_prec = cond_experiment(Split, splitpcg_precisions, 1e-4 , kappa_range, 150)
