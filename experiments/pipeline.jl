using MixedPrecisionPCG
using .NumericalExperiments
using Random

Random.seed!(1234)

function runpcgexperiments(
    scheme    ::Type{<:PreconditioningScheme},
    v_ls      ::Vector{LinearSystem{T}},
    v_prec    ::Vector{<:AbstractMatrix},
    v_iter    ::Vector{Int},
    precisions::AbstractVector) where T <: AbstractFloat

    n_ls = length(v_ls)

    v_ad = Vector{AccuracyData{Float64}}(undef, n_ls)

    for i = 1:n_ls

        v_ad[i] = collect_data(scheme, precisions, v_ls[i], v_prec[i], v_iter[i])

    end

    return v_ad
end


# System 1
# =============================================================================

n1 = 40
Y = rand(n1, n1)

# Compute random orthogonal matrix.
Q = qr(Y).Q 

kappa = 10^5

# Generate geometrically distributed eigenvalues.
D = diagm([ kappa^((i - 1) / (n1 - 1)) for i in 1:n1 ])

# Construct matrix with prescribed eigenvalues.
A1 = sparse(Hermitian(Q * D * Q'))

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
L1 = sparse( cholesky(Float16.(A1) + 10.5 .* I(n1) ).L )

# System 2
# =============================================================================

# Construct SPD, banded coefficient matrix.
A2 = mat"delsq(numgrid('S', 102))"

n2 = size(A2, 1)

b2 = ones(size(A2, 1))

x2 = A2 \ b2

x0_2 = zeros(n2)

max_iter2 = 100

ls2 = LinearSystem(A2, b2, x2, x0_2)

L2  = mat"ichol($A2)"


# System 3: System 2 with a different preconditioner.
# =============================================================================

L3 = mat"ichol($A2, struct('michol', 'on'))"

# Set precisions
d = Float64
s = Float32
h = Float16

splitpcg_precisions = [(d, d), (d, s), (s, d), (s, s), (d, h)]
leftpcg_precisions  = [d, s, h]

v_ls       = [ls1, ls2, ls2]
v_prec     = [L1, L2, L3]
iterations = [max_iter1, max_iter2, max_iter2]

v_ad_left  = runpcgexperiments(Left,  v_ls, v_prec, iterations, leftpcg_precisions)
v_ad_split = runpcgexperiments(Split, v_ls, v_prec, iterations, splitpcg_precisions)

