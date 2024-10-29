using MATLAB, Plots

"""
Reproducing example from 
"""

# Example data.
data_path = pwd() * "/experiments/data/proZdenka.mat"
mf = MatFile(data_path)  # Load .mat file.
A  = get_variable(mf, "Afree") # Get matrix.
b  = get_variable(mf, "bfree") # Get right-hand side.
x  = A\b                       # Solve system.
close(mf)

# Construct left preconditioners.
L1      = mat"ichol($A, struct('shape', 'lower'))"
M1      = FactorizationPreconditioner{Float64, Left}(L1, transpose(L1))
droptol = 1e-2
L2      = mat"ichol($A, struct('type', 'ict', 'droptol', $droptol, 'shape', 'lower'))"
M2      = FactorizationPreconditioner{Float64, Left}(L2, transpose(L2))
M3      = FactorizationPreconditioner{Float32, Left}(L2, transpose(L2))
M4      = FactorizationPreconditioner{Float64, Float64, Split}(L2, transpose(L2))


# Problem data
n        = size(A, 1)
max_iter = 50
cd1      = ConvergenceData{Float64}(n, max_iter)
cd2      = ConvergenceData{Float64}(n, max_iter)
cd3      = ConvergenceData{Float64}(n, max_iter)
cd4      = ConvergenceData{Float64}(n, max_iter)
x0       = zeros(n)                              # Initial guess.


function run_experiment()

    left_pcg!( cd1, A, M1, b, x0, max_iter)
    left_pcg!( cd2, A, M2, b, x0, max_iter)
    left_pcg!( cd3, A, M3, b, x0, max_iter)
    split_pcg!(cd4, A, M4, b, x0, max_iter)

    return cd1, cd2, cd3, cd4

end

function plot_convergence(v_cd::Vector{ConvergenceData{Float64}})

    for k in 1:length(v_cd)
        compute_error_norm!(v_cd[k], x, A)
    end

    plot(collect(1:v_cd[1].iter_number), v_cd[1].relative_error_norm, yscale = :log10, label = "cd 1")

    for k in 2:length(v_cd)
        display(plot!(collect(1:v_cd[k].iter_number), v_cd[k].relative_error_norm))
    end

    #labels = permutedims([ "cd " * string(i) for i in 1:length(v_cd) ])
end

