using MixedPrecisionPCG
using LinearAlgebra, MATLAB, Random

Random.seed!(1234)


# System 1
# =============================================================================

n1    = 40
kappa1 = 10^5

# Construct matrix with prescribed eigenvalues.
A1 = geomdist_eigvalmatrix(kappa1, n1)

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

max_iter2 = 200

ls2 = LinearSystem(A2, b2, x2, x0_2)

# Construct preconditioner using MATLAB routine ichol (incomplete Cholesky).
L2  = mat"ichol($A2)"

# System 3: System 2 with a different preconditioner that accelerates convergence.
# =============================================================================

# Generate preconditioner using a modified incomplete Cholesky factorization.
L3 = mat"ichol($A2, struct('michol', 'on'))"


# System 4: Dense matrix.
# =============================================================================

function randsvd_spd(n, mode)

    A = mat"gallery('randsvd', [$n,$n], 1e12, $mode);"
    sva = svd(A).S;
    V   = qr(rand(n,n)).Q;
    A  = V * diagm(sva) * V';
    A  = 0.5 * (A + A');

    return A

end

n4   = 500;
A4   = randsvd_spd(n4, 1) # Fast singular value decay.
b4   = rand(n4);
L4   = low_precision_preconditioner(A4, tol=1e-4);
x4   = A4 \ b4;
x0_4 = zeros(n4);

kappa4   = cond(Matrix(A4));
kappaM4 = cond(Matrix(L4 * L4'));

max_iter4 = 300;

ls4  = LinearSystem(A4, b4, x4, x0_4);

A5  = randsvd_spd(n4, 2) # Slow singular value decay.
L5   = low_precision_preconditioner(A5, tol=1e-4);
x5  = A5 \ b4;

ls5 = LinearSystem(A5, b4, x5, x0_4)


# Set precisions
d = Float64
s = Float32
h = Float16

# Set unit roundoff values
ud = 0.5 * eps(d)
us = 0.5 * eps(s)
uh = 0.5 * eps(h)

splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s), (d, h)]
leftpcg_precisions  = [d, s, h]

v_ls       = [      ls1,       ls2,        ls4,       ls5]
v_prec     = [       L1,        L2,         L4,        L5]
iterations = [max_iter1, max_iter2,  max_iter4, max_iter4]

v_ad_left      = runpcgexperiments(Left,      v_ls, v_prec, iterations, leftpcg_precisions)
v_ad_split     = runpcgexperiments(Split,     v_ls, v_prec, iterations, splitpcg_precisions)
v_ad_saadsplit = runpcgexperiments(SaadSplit, v_ls, v_prec, iterations, splitpcg_precisions)

#splitpcg_precisions = [(s, d)]
kappa_range         = 10.0 .^collect(2:2:10)

#ad_vec, kappa_range_prec = cond_experiment(Split, splitpcg_precisions, 1e-4 , kappa_range, 150)
