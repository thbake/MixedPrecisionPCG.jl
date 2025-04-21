
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
