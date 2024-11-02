using MATLAB, Plots, LaTeXStrings, MixedPrecisionPCG

export run_experiment!

setprecision(BigFloat, 128)

"""
Reproducing example from https://doi.org/10.1016/j.laa.2024.04.003

Thanks to Jan Papez for providing me with the data.
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
#M1      = FactorizationPreconditioner{Float64, Left}(L1, transpose(L1))
droptol = 1e-2
L2      = mat"ichol($A, struct('type', 'ict', 'droptol', $droptol, 'shape', 'lower'))"

# Problem data
n        = size(A, 1)
max_iter = 50
x0       = zeros(n)


"""
Generate preconditioners data structures based on preconditioning scheme, a 
vector of precisions, and the actual preconditioners.
"""
function generate_preconditioners(
    scheme    ::Type{<:PreconditioningScheme},
    Pl        ::AbstractMatrix,
    Pr        ::AbstractMatrix,
    precisions::Vararg{Type{<:AbstractFloat}, N}) where N

    preconditioners = [ FactorizationPreconditioner{u, scheme}(Pl, Pr) for u in precisions]
    

    return preconditioners
end

function generate_preconditioners(
    Pl        ::AbstractMatrix,
    Pr        ::AbstractMatrix,
    precisions::Vararg{Tuple{Type, Type}, N}) where {N}

    preconditioners = [ FactorizationPreconditioner{uL, uR, Split}(Pl, Pr) for (uL, uR) in precisions]
    

    return preconditioners
end

function generate_convergence_data(n::Int, max_iter::Int, length::Int)

    return [ ConvergenceData{Float64}(n, max_iter) for _ in 1:length ]

end



get_pcg_variant(::Left)  = left_pcg!
get_pcg_variant(::Split) = split_pcg!



function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    left_precision = getprecisions(preconditioner)

    return L"$u_L = $" * string(left_precision)  

end

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Split}) where {uL, uR}

    left_precision, right_precision = getprecisions(preconditioner)
    return  L"$u_L = $" * string(left_precision) * L", $u_R = $" * string(right_precision) 

end

gettitle(::Left)  = "Mixed precision left PCG"
gettitle(::Split) = "Mixed precision split PCG"

function plot_convergence(
    v_cd  ::Vector{ConvergenceData{Float64}},
    v_prec::Vector{<:AbstractPreconditioner},
    scheme::Type{<:PreconditioningScheme})

    # Compute error in the A-norm for all runs.
    for k in eachindex(v_cd)
        compute_error_norm!(v_cd[k], x, A)
    end

    p = plot(title = gettitle( scheme() ) )

    for k in eachindex(v_cd)
        display(
            plot!( 
                collect(1:v_cd[k].iter_number), v_cd[k].relative_error_norm, yscale = :log10, label = getlabel(v_prec[k])
            ) 
        )
    end

end

function run_experiment!(scheme::Type{<:PreconditioningScheme})

    f      = get_pcg_variant(scheme())
    #v_prec = generate_preconditioners(scheme, L2, transpose(L2), Float64, Float32, Float16)
    v_prec = generate_preconditioners(L2, transpose(L2), (Float64, Float64), (Float64, Float32), (Float16, Float16))
    #v_prec = generate_preconditioners(scheme, I(n), I(n), Float64, Float32, Float16)
    v_cd   = generate_convergence_data(n, max_iter, length(v_prec))

    for i in eachindex(v_cd)
        f( v_cd[i], A, v_prec[i], b, x0, max_iter )
    end

    plot_convergence(v_cd, v_prec, scheme)

end

