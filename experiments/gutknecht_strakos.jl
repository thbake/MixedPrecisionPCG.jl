using Plots
using Random
using MixedPrecisionPCG
using MATLAB
using IterativeSolvers

Random.seed!(123)

n = 48

ω = ones(n)

psi1 = 10
psieven = 0.01
psiodd  = 100
psi45   = 1e-2
psi46   = 1e-3
psi47   = 1e-4


f(x) = (x % 2) == 0 ? psieven : psiodd 

Ψ = cat([psi1], [f(i) for i in 2:44], [psi45, psi46, psi47], dims=1) 

T = zeros(n,n)

T[1,1] = 1 

for i in 2:n

    T[i, i]     = 1 + Ψ[i - 1]

    T[i, i - 1] = sqrt(Ψ[i - 1])

    T[i - 1, i] = sqrt(Ψ[i - 1])

end

V = qr(rand(n,n)).Q
A = Hermitian(V * T * V')
#A = Hermitian(V * diagm(Float16.(eigvals(A))) * V')
x0 = zeros(n)
b  = V[:, 1]
x  = A\b

pertA = 

max_iter = 120

# Initialize convergence data structure
cd      = ConvergenceData{Float64}(n, max_iter)
left_cd = ConvergenceData{Float64}(n, max_iter)
split_cd = ConvergenceData{Float64}(n, max_iter)

A_sparse    = sparse(A)
alpha_shift = maximum(sum( abs, A, dims = 2)./diag(A)) - 2

droptol = 1e-3
L2      = mat"ichol($A_sparse, struct('type', 'ict', 'droptol', $droptol, 'diagcomp', $alpha_shift))"
L = cholesky(A).L

kappaL = cond(Matrix(L2))
kappaA = cond(A)

inverseprec = inv(Matrix(L2))
precA  = inverseprec' * inverseprec * A
kappaAprec = cond(precA)

println("κL = ", kappaL, "\nκA = ", kappaA, "\nκprecA = ", kappaAprec )

ML = FactorizationPreconditioner{Float64, Left}(L2, L2')
MS = FactorizationPreconditioner{Float64, Float64, Split}(L2, L2')

hscg!(cd,            A_sparse,     b, x0, max_iter)
left_pcg!(left_cd,   A_sparse, ML, b, x0, max_iter)
split_pcg!(split_cd, A_sparse, MS, b, x0, max_iter)



# A SPD ==> Its two norm is eigenvalue of largest magnitude.
normA = opnorm(A)
normx = norm(A\b)
relative_norm_denominator = inv(normA * normx)


# Compute norms
updated_resnorms       = [ norm(@view cd.updated_residuals[:, j])       for j in 1:max_iter ] 
left_updated_resnorms  = [ norm(@view left_cd.updated_residuals[:, j])  for j in 1:max_iter ] 
split_updated_resnorms = [ norm(@view split_cd.updated_residuals[:, j]) for j in 1:max_iter ] 

compute_residual_norm!(cd,       A_sparse, b) 
compute_residual_norm!(left_cd,  A_sparse, b) 
compute_residual_norm!(split_cd, A_sparse, b) 

true_resnorms       = cd.true_residuals
left_true_resnorms  = left_cd.true_residuals
split_true_resnorms = split_cd.true_residuals

# Divide by ||A||*||x||
updated_resnorms      = updated_resnorms       .* relative_norm_denominator
true_resnorms         = true_resnorms          .* relative_norm_denominator

left_updated_resnorms = left_updated_resnorms  .* relative_norm_denominator
left_true_resnorms    = left_true_resnorms     .* relative_norm_denominator

split_updated_resnorms = split_updated_resnorms .* relative_norm_denominator
split_true_resnorms    = split_true_resnorms    .* relative_norm_denominator


generate_yticks(minval, maxval) = 10.0.^collect(minval:2:maxval + 1)
yticks = generate_yticks(1, 21)

default(
    yscale     = :log10,
    markersize = 2,
    alpha      = 0.8, 
    #yticks     = generate_yticks(1, 21)
)

# Plot norms of the residual
plot(updated_resnorms, yscale = :log10, label = "Updated residual norm", marker = :circle)
plot!(true_resnorms, label = "True residual norm", marker = :circle)

#plot!(left_updated_resnorms,   label = "Left updated residual norm", alpha = 1.0, marker = :diamond)
#plot!(left_true_resnorms,      label = "Left true residual norm", marker = :diamond)
plot!(split_updated_resnorms,  label = "Split updated residual norm", alpha = 1.0, marker = :star)
plot!(split_true_resnorms,     label = "Split true residual norm", marker = :star)

# Compute norms of the error.

error       = compute_error!(cd,       x, A_sparse)
left_error  = compute_error!(left_cd,  x, A_sparse)
split_error = compute_error!(split_cd, x, A_sparse)

# Plot norms of the error.
plot(cd.relative_error_norm,        label = "Error")
plot!(left_cd.relative_error_norm,  label = "Left error")
plot!(split_cd.relative_error_norm, label = "Split error")
