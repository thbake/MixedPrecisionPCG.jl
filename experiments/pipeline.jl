using MixedPrecisionPCG
using .NumericalExperiments

# System 1
# ========

n = 40
Y = rand(n, n)

# Compute random orthogonal matrix.
Q = qr(Y).Q 

kappa = 10^5

# Generate geometrically distributed eigenvalues.
D = diagm([ kappa^((i - 1) / (n - 1)) for i in 1:n ])

# Construct matrix with prescribed eigenvalues.
A        = Hermitian(Q * D * Q')
A_sparse = sparse(A) 

# Choose random solution.
x = rand(n)

# Get right-hand side.
b = A_sparse * x

# Set initial guess to zero
x0 = zeros(n)

# Maximal number of iterations
max_iter = 150

ls = LinearSystem(A_sparse, b, x, x0)

d = Float64
s = Float32
h = Float16

splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s), (d, h)]
leftpcg_precisions  = [d, s, h]

# Create low precision Cholesky factorization as preconditioner
L = cholesky(Float16.(A) + 10.5 .* I(n) ).L 


ad_split = collect_data(Split, splitpcg_precisions, ls, L)
ad_left  = collect_data(Left,   leftpcg_precisions, ls, L)
