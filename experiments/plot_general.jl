using Plots, LaTeXStrings

export plot_accuracy_data, condition_plot

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    left_precision = getprecisions(preconditioner)

    return L"$u_L = $" * string(left_precision)  

end

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Split}) where {uL, uR}

    left_precision, right_precision = getprecisions(preconditioner)
    return  L"$u_L = $" * string(left_precision) * L", $u_R = $" * string(right_precision) 

end

function getlabels(precisions::AbstractVector, label_dict, ::Split)

    labels = Vector{String}(undef, length(precisions))

    i = 1

    for (prec1, prec2) in precisions

        
        label = "(" * label_dict[prec1] * "," * label_dict[prec2] * ")"

        labels[i] = label

        i += 1

    end

    return permutedims(labels)

end

function getlabels(precisions::AbstractVector, label_dict, ::Left)

    labels = Vector{String}(undef, length(precisions))

    for i in eachindex(precisions)

        labels[i] = label_dict[precisions[i]]

    end

    return permutedims(labels)
end


gettitle(::Left)  = "Mixed precision left PCG"
gettitle(::Split) = "Mixed precision split PCG"



function plot_accuracy_data(
    ad        ::AccuracyData,
    precisions::AbstractVector,
    scheme    ::Type{<:PreconditioningScheme})

    label_dict = Dict(
        Float64 => "d",
        Float32 => "s",
        Float16 => "h"
    )

    labels = getlabels(precisions, label_dict, scheme())

    default(
        yscale     = :log10, 
        legend     = :topright,
        markersize = 2,
        alpha      = 0.8,
        label      = labels
    )

    p1 = plot(
        ad.trueresnorm,
        ylabel     = L"$\frac{||b - Ax_k||}{||A|| ||x||}$",
        
    )

    p2 = plot(
        ad.updatedresnorm,
        ylabel          = L"$\frac{||r_k||}{||A|| ||x||}$",
    )

    p3 = plot(
        ad.errornorm,
        ylabel = L"$\frac{||x - x_k||_A}{||x - x_0||_A}$",
    )

    p4 = plot(
        ad.resgapnorm,
        ylabel        = L"$\frac{||b - Ax_k - M_L \hat{r}_k||}{||A|| ||x||}$"
    )

    plot(p1, p2, p3, p4, layout = (2,2), title = "My plots")
end

function condition_plot(
    ad_vec     ::Vector{AccuracyData},
    kappa_range::Vector{Float64},
    precision  ::Type{<:AbstractFloat})

    default(
        yscale   = :log10,
        legend   = :topright,
        ylabel   = L"$\frac{||b - A x_k ||}{||A|| ||x||}$"
    )

    p = plot()

    colors = palette(:Dark2_5)

    for i in eachindex(kappa_range)

        label = raw"$\kappa(M_L) = 10^{" * string( log10(kappa_range[i]) ) * raw"}$" 

        # First (and in this case only) entry in the dictionary.
        max_ratio, max_ratio_idx = ad_vec[i].max_ratios[1]

        uL_kappa = 0.5 * eps(precision) * kappa_range[i] * max_ratio

        color = colors[i]

        plot!(
            ad_vec[i].trueresnorm,
            label   = label,
            lc      =  color
        )

        scatter!(
            [uL_kappa], 
            marker      = :xcross,
            markercolor = color,
            label       = ""
        )

        scatter!(
            [max_ratio_idx], 
            [1.0],
            marker      = :cross,
            markercolor = color,
            label       = "",
        )


    end

    display(p)

end

#scatter!([idx[2]], [1.0], marker = :xcross, markercolor = :black, label = L"$max_j ||x_j||$")
