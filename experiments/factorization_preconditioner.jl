using MATLAB  
using MixedPrecisionPCG

export run_experiment!

#setprecision(BigFloat, 128)

"""
Reproducing example from https://doi.org/10.1016/j.laa.2024.04.003

Thanks to Jan Papez for providing me with the data.
"""

function load_data()
    # Example data.
    data_path = pwd() * "/experiments/data/proZdenka.mat"
    mf = MatFile(data_path)  # Load .mat file.
    A  = get_variable(mf, "Afree") # Get matrix.
    b  = get_variable(mf, "bfree") # Get right-hand side.
    x  = A\b                       # Solve system.
    close(mf)

    return A, x, b
end

A, x, b = load_data()

# Construct left preconditioners.
L1      = mat"ichol($A, struct('shape', 'lower'))"
#M1      = FactorizationPreconditioner{Float64, Left}(L1, transpose(L1))
droptol = 1e-2
L2      = mat"ichol($A, struct('type', 'ict', 'droptol', $droptol, 'shape', 'lower'))"

# Problem data
n        = size(A, 1)
max_iter = 50
x0       = zeros(n)



function run_experiment!(scheme::Type{<:PreconditioningScheme})

    pcg_variant = get_pcg_variant(scheme())
    v_prec      = generate_preconditioners(scheme, L2, transpose(L2), Float64, Float32, Float16)
    v_cd        = initialize_convergence_data(n, max_iter, length(v_prec))


    for i in eachindex(v_cd)
        pcg_variant( v_cd[i], A, v_prec[i], b, x0, max_iter )
    end

    #compute_errors!(v_cd, x, A, b, v_prec, scheme)
    compute_errors!(v_cd, x, A, b, scheme)

    plot_convergence(v_cd, v_prec, scheme, A)

end

    #v_prec      = generate_preconditioners(scheme, I(n), I(n), Float64, Float32, Float16)
    #v_prec     = generate_preconditioners(L2, transpose(L2), (Float64, Float64), (Float64, Float32), (Float32, Float64), (Float32, Float32), (Float16, Float16))
    #v_prec = generate_preconditioners(scheme, I(n), I(n), Float64, Float32, Float16)
