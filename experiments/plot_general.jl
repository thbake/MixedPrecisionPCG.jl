using Plots, LaTeXStrings

export plot_convergence

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    left_precision = getprecisions(preconditioner)

    return L"$u_L = $" * string(left_precision)  

end

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Right}) where {uL, uR}

    right_precision = getprecisions(preconditioner)

    return L"$u_L = $" * string(right_precision)  

end

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Split}) where {uL, uR}

    left_precision, right_precision = getprecisions(preconditioner)
    return  L"$u_L = $" * string(left_precision) * L", $u_R = $" * string(right_precision) 

end

get_Anormerror(   v_cd::Vector{ConvergenceData{Float64}}) = [ v_cd[k].relative_error_norm     for k in 1:length(v_cd)]
get_backwarderror(v_cd::Vector{ConvergenceData{Float64}}) = [ v_cd[k].relative_backward_error for k in 1:length(v_cd)]

gettitle(::Left)  = "Mixed precision left PCG"
gettitle(::Split) = "Mixed precision split PCG"
gettitle(::Right) = "Mixed precision right PCG"

function plot_convergence(
    v_cd  ::Vector{ConvergenceData{Float64}},
    v_prec::Vector{<:AbstractPreconditioner},
    scheme::Type{<:PreconditioningScheme},
    A     ::AbstractMatrix)


    title = plot(title = gettitle( scheme() ) )

    iter_number         = v_cd[1].iter_number
    Anormerror_data     = get_Anormerror(v_cd)
    backwarderror_data  = get_backwarderror(v_cd)
    labels              = permutedims([getlabel(v_prec[k]) for k in 1:length(v_prec)])
    yfontsize           = font(8)


    println(typeof(v_prec[1]))
    #condition_numbers = [preconditioned_condition_number(A, v_prec[i]) for i in eachindex(v_prec)]


    Anormerror_plot = plot(
        collect(1:iter_number),
        Anormerror_data,
        yscale = :log10,
        label  = labels,
        ylabel = "relative error in the A-norm",
        yguidefont = yfontsize)
        #legendfontsize= 7)
    

    backwarderror_plot = plot(
        collect(1:iter_number),
        backwarderror_data,
        yscale = :log10,
        label  = labels,
        ylabel = "relative backward error",
        yguidefont = yfontsize)

    plot(Anormerror_plot, backwarderror_plot, layout = (1,2), suptitle = gettitle( scheme() ), legendfontsize = 5, plot_titlefontsize = 7 )

end
