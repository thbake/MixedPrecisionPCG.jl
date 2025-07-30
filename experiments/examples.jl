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

max_iter2 = 150

ls2 = LinearSystem(A2, b2, x2, x0_2)

# Construct preconditioner using MATLAB routine ichol (incomplete Cholesky).
L2  = mat"ichol($A2)"

# System 3: System 2 with a different preconditioner that accelerates convergence.
# =============================================================================

# Generate preconditioner using a modified incomplete Cholesky factorization.
L3 = mat"ichol($A2, struct('michol', 'on'))"


# System 4: Dense matrix.
# =============================================================================

function randsvd_spd(n, mode, cutoff, α = 0.0)

    A = mat"gallery('randsvd', [$n,$n], 1e8, $mode);"
    sva = svd(A).S;
    V   = qr(rand(n,n)).Q;
    A  = V * diagm(sva) * V';
    A  = 0.5 * (A + A');

    prec_eigvals = vcat(sva[1:cutoff], [sva[cutoff] for _ in 1:n-cutoff])

    M = V * diagm(prec_eigvals) * V'
    M = 0.5 * (M + M') 

    M[n,n] += α

    #L = low_precision_preconditioner(M, tol = 1e-4) 
    L = cholesky(M).L

    return A, L

end

n4   = 300;
#A4   = randsvd_spd(n4, 1) # Fast singular value decay.
b4   = rand(n4);
#L4   = low_precision_preconditioner(A4, tol=1e-4) + I(n4);

A4, L4 = randsvd_spd(n4, 3, 160)
#L4 = I(n4);
x4   = A4 \ b4;
x0_4 = zeros(n4);

kappa4   = cond(Matrix(A4));
kappaM4 = cond(Matrix(L4 * L4'));

max_iter4 = 2000;

ls4  = LinearSystem(A4, b4, x4, x0_4);

#A5  = randsvd_spd(n4, 2) # Slow singular value decay.
#L5  = low_precision_preconditioner(A5, tol=1e-4) + I(n4);
A5, L5 = randsvd_spd(n4, 2, n4)
x5  = A5 \ b4;

ls5 = LinearSystem(A5, b4, x5, x0_4)

# Example 6: Strakos Matrix with eigenvalues accumulated to the left.
# ==================================================================

strakos_mat(n, l1, ln, rho) = diagm(vcat([l1], [l1 + (i - 1)/(n - 1) * (ln - l1) * rho^(n - i) for i in 2:n-1], [ln]))

function mat_prec(n::Int, l1::Float64, ln::Float64, rho::Float64, cutoff::Int)

   A = strakos_mat(n, l1, ln, rho)

   eigs = diag(A)

   #V   = qr(rand(n,n)).Q;

   # Truncate first n - "cutoff" eigenvalues <=> Preserve first "cutoff" eigenvalues.
   prec_eigvals = vcat(eigs[1:cutoff], [eigs[cutoff] for _ in 1:n-cutoff])

   M = diagm(prec_eigvals)

   L = cholesky(M).L

   println(cond((L \ M) / L'))

   return A, L

end

n6        = 85
A6, L6    = mat_prec(n6, 1.0, 1e+5, 0.6, 1)
M6        = Symmetric(L6 * L6')
b6        = inv(sqrt(n6)) .* ones(n6)
x6        = A6\b6
max_iter6 = 500

splitprecA = (L6 \ A6) / (L6')
leftprecA  = M6 \ A6

ls6 = LinearSystem(A6, b6, x6, zeros(n6))


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

#v_ls       = [      ls1,       ls2,        ls4,       ls5]
#v_prec     = [       L1,        L2,         L4,        L5]
#iterations = [max_iter1, max_iter2,  max_iter4, max_iter4]

#v_ls       = [      ls1,       ls2,        ls4,       ls5]
#v_prec     = [       L1,        L2,         L4,        L5]
#iterations = [max_iter1, max_iter2,  max_iter4, max_iter4]

v_ls       = [      ls1,        ls4,       ls6]
v_prec     = [       L1,         L4,        L6]
iterations = [max_iter1,  max_iter4, max_iter6]

#v_ls       = [      ls4,       ls5]
#v_prec     = [       L4,        L5]
#iterations = [max_iter4, max_iter4]

v_ad_left      = runpcgexperiments(Left,      v_ls, v_prec, iterations, leftpcg_precisions)
v_ad_split     = runpcgexperiments(Split,     v_ls, v_prec, iterations, splitpcg_precisions)
v_ad_saadsplit = runpcgexperiments(SaadSplit, v_ls, v_prec, iterations, splitpcg_precisions)

#splitpcg_precisions = [(s, d)]
kappa_range         = 10.0 .^collect(2:2:10)

#ad_vec, kappa_range_prec = cond_experiment(Split, splitpcg_precisions, 1e-4 , kappa_range, 150)
