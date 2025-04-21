using Plots
using LaTeXStrings
using Random
using MixedPrecisionPCG


Random.seed!(123)

setprecision(BigFloat, 128)

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

# Initialize convergence data structure
cd          = ConvergenceData{Float64}(n, max_iter)
left_cd     = ConvergenceData{Float64}(n, max_iter)
split_cd_dd = ConvergenceData{Float64}(n, max_iter)
split_cd_ds = ConvergenceData{Float64}(n, max_iter)
split_cd_sd = ConvergenceData{Float64}(n, max_iter)
split_cd_ss = ConvergenceData{Float64}(n, max_iter)
split_cd_dh = ConvergenceData{Float64}(n, max_iter)


alpha_shift = maximum(sum( abs, A, dims = 2)./diag(A)) - 2

droptol = 1e-3

# Compute incomplete cholesky factorization.
#L2 = mat"ichol($A_sparse, struct('type', 'ict', 'droptol', $droptol, 'shape', 'lower'))"
L2       = mat"ichol($A_sparse, struct('type', 'ict', 'droptol', $droptol, 'diagcomp', $alpha_shift))"
L2_dense = Matrix(L2)
invL2    = inv(L2_dense)


L    = cholesky(Float16.(A) + 10.5 .* I(n) ).L 



# Initialize preconditioner
#ML = FactorizationPreconditioner{Float64, Left}(L2, L2')
#MS = FactorizationPreconditioner{Float64, Float64, Split}(L2, L2')
MLd = FactorizationPreconditioner{Float64, Left}(L, L')
MLs = FactorizationPreconditioner{Float32, Left}(L, L')

MSdd = FactorizationPreconditioner{Float64, Float64, Split}(L, L')
MSds = FactorizationPreconditioner{Float64, Float32, Split}(L, L')
MSsd = FactorizationPreconditioner{Float32, Float64, Split}(L, L')
MSss = FactorizationPreconditioner{Float32, Float32, Split}(L, L')
MSdh = FactorizationPreconditioner{Float64, Float16, Split}(L, L')


# Compute preconditioned matrix.
precA = invL2' * (invL2 * A)

hscg!(           cd, A_sparse, b, x0, max_iter)
left_pcg!(  left_cd, A_sparse, MLd, b, x0, max_iter)
split_pcg!(split_cd_dd, A_sparse, MSdd, b, x0, max_iter)
split_pcg!(split_cd_ds, A_sparse, MSds, b, x0, max_iter)
split_pcg!(split_cd_sd, A_sparse, MSsd, b, x0, max_iter)
split_pcg!(split_cd_ss, A_sparse, MSss, b, x0, max_iter)
split_pcg!(split_cd_dh, A_sparse, MSdh, b, x0, max_iter)

# Form preconditioned matrices
leftprec_A  = L' \ (L \ A_sparse)
splitprec_A = (L \ A_sparse)

# A SPD ==> Its two norm is eigenvalue of largest magnitude.
normA = opnorm(A)
normx = norm(x)

relativenorm_denominator = inv(normA * normx)


# Compute norms of true residuals
compute_residual_norm!(cd,       A_sparse, b) 
compute_residual_norm!(left_cd,  A_sparse, b)

# Split true residual norms
compute_residual_norm!(split_cd_dd, A_sparse, b)
compute_residual_norm!(split_cd_ds, A_sparse, b)
compute_residual_norm!(split_cd_sd, A_sparse, b)
compute_residual_norm!(split_cd_ss, A_sparse, b)
compute_residual_norm!(split_cd_dh, A_sparse, b)


apply_norm(matrix, preconditioner) = [ norm( preconditioner * @view(matrix[:, j])) for j in 1:max_iter ]

apply_norm(matrix)                 = [ norm( @view(matrix[:, j])) for j in 1:max_iter ]

# Prepare data for plotting
# =========================

# Relative residual norm
# ----------------------------
reltrueres = [
    split_cd_dd.true_residuals,
    split_cd_ds.true_residuals,
    split_cd_sd.true_residuals,
    split_cd_ss.true_residuals,
    split_cd_dh.true_residuals
] .* relativenorm_denominator

# Norm of (unpreconditioned) updated residuals
# --------------------------------------------
updatedresnorms = map(apply_norm, [
    split_cd_dd.updated_residuals,
    split_cd_ds.updated_residuals,
    split_cd_sd.updated_residuals,
    split_cd_ss.updated_residuals,
    split_cd_dh.updated_residuals
], MSdd.Pl) .* relativenorm_denominator

# Norms of residual gap divided by ||A|| ||x||
# ----------------------------------------------------------
residual_gap_norms = [
    relative_residual_gap(A, L, b, x, split_cd_dd, max_iter),
    relative_residual_gap(A, L, b, x, split_cd_ds, max_iter),
    relative_residual_gap(A, L, b, x, split_cd_sd, max_iter),
    relative_residual_gap(A, L, b, x, split_cd_ss, max_iter),
    relative_residual_gap(A, L, b, x, split_cd_dh, max_iter),
]


val, idx   = findmax(norm.(split_cd_ss.iterates .* norm(x)))  # Get maximum norm of iterates

# Plotting 
# =========================================
default(
    yscale     = :log10,
    yticks     = 10.0.^collect(-25:5:1),
    markersize = 2,
    alpha      = 0.8, 
    legend     = :topright
    #yticks     = generate_yticks(1, 21)
)


labels     = ["(d,d)", "(d,s)", "(s,d)", "(s,s)", "(d,h)"]

p1 = plot(
    reltrueres,      
    label      = permutedims(labels),
    title      = "Split PCG",
    ylabel     = L"$\frac{||b - Ax_k||}{||A|| ||x||}$",
    linestyle  = :auto)

p2 = plot(
    updatedresnorms,
    label           = permutedims(labels),
    title           = "Split PCG", 
    ylabel          = L"$\frac{||r_k||}{||A|| ||x||}$",
    yticks          = :auto)


p3 = plot(
    residual_gap_norms,
    label              = permutedims(labels),
    title              = "Split PCG",
    ylabel             = L"$\frac{||b - Ax_k - M_L \hat{r}_k||}{||A|| ||x||}$")

plot(p1, p2, layout = (2, 1), title = "Split PCG")


# Compute norms of the error.

error       = compute_error!(cd,       x, A_sparse)
left_error  = compute_error!(left_cd,  x, A_sparse)
split_error = compute_error!(split_cd_dd, x, A_sparse)

compute_backward_error!(cd, Matrix(A_sparse), b)


function relative_residual_gap_bound(S, x0, x, uL, uA, u, kappaML)


end


# Plot norms of the error.
#p1 = plot(cd.relative_error_norm,        label = "A-norm error")
#plot!(cd.relative_backward_error, label = "BE")
#plot!(left_cd.relative_error_norm,  label = "Left error")
#plot!(split_cd.relative_error_norm, label = "Split error")


scatter!([idx[2]], [1.0], marker = :xcross, markercolor = :black, label = L"$max_j ||x_j||$")

#scatter!(idx[2], 0, xticks = 1:max_iter, yticks = -1:1)

#l = @layout[a; b]
#
#plot(p1, p2)






