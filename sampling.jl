# module TripletSampling

using Plots
using Distributions
using TripletEmbeddings

import RecipesBase: plot

function Embedding(d::Int, n::Int, classes::Int; σ::Real=1.0, bias::Vector{Float64}=5*ones(d))
    d == length(bias) || throw(DimensionMismatch("d and length(bias) must be equal."))
    
    R = [[cos(θ) -sin(θ); sin(θ) cos(θ)] for θ in rand(Uniform(0, 2π), classes)]

    X = [broadcast(+, σ * randn(d, n), (c-1) * R[c] * bias) for c in 1:classes]
    return [TripletEmbeddings.Embedding(X[c]) for c in 1:classes]
end

function Embedding(d::Int, n::Vector{Int}; σ::Real=1.0, bias::Vector{Float64}=5*ones(d))
    d == length(bias) || throw(DimensionMismatch("d and length(bias) must be equal."))
    classes = length(n)
    
    R = [[cos(θ) -sin(θ); sin(θ) cos(θ)] for θ in rand(Uniform(0, 2π), classes)]
    X = [broadcast(+, σ * randn(d, n[c]), (c-1) * R[c] * bias) for c in 1:classes]
    return [TripletEmbeddings.Embedding(X[c]) for c in 1:classes]
end

# function core(n::Vector{Int})
#     classes == length(n) || throw(DimensionMismatch("classes != length(n)"))

#     for c in 1:classes
#         c_k = mod(c, classes) + 1
#         n_ij = n[c]   # i is always in the same class as j
#         n_k = n[c_k] # k never is in the same class as i or j
#         for k = 1:n_k, j = 1:n_ij, i = 1:n_ij
#             if i != j
#                 t_i = c > 1 ? sum(n[1:c-1]) + i : i
#                 t_j = c > 1 ? sum(n[1:c-1]) + j : j
#                 t_k = c_k > 1 ? sum(n[1:c]) + k : k 
#                 @show (c, c_k, t_i, t_j, t_k, n)
#             end
#         end
#     end    
# end

function TripletEmbeddings.Triplets(X::Vector{TripletEmbeddings.Embedding{Float64}})
    n = nitems.(X)
    classes = length(X)
    all(y -> y == ndims(X[1]), ndims.(X)) || throw(DimensionMismatch("All embeddings must have the same dimension"))

    triplets = Vector{Tuple{Int,Int,Int}}(undef, sum(n) * binomial(sum(n) - 1, 2))
    counter = 0

    for c in 1:classes
        c_k = mod(c, classes) + 1
        n_ij = n[c]   # i is always in the same class as j
        n_k = n[c_k] # k never is in the same class as i or j
        for k = 1:n_k, j = 1:n_ij, i = 1:n_ij
            if i != j
                counter += 1
                t_i = c > 1 ? sum(n[1:c-1]) + i : i
                t_j = c > 1 ? sum(n[1:c-1]) + j : j
                t_k = c_k > 1 ? sum(n[1:c]) + k : k 
                @inbounds triplets[counter] = (t_i, t_j, t_k)
            end
        end
    end    

    return Triplets(triplets[1:counter])
end

function RecipesBase.plot(X::Vector{TripletEmbeddings.Embedding{Float64}}; kwargs...)
    plot()
    foreach(x -> scatter!(x[1,:], x[2,:]; kwargs...), X)
end

function fit(loss::TripletEmbeddings.AbstractLoss, triplets::Triplets, d::Int)
    d > 0 || throw(ArgumentError("dimension d must be > 0"))

    X = TripletEmbeddings.Embedding(d, maximum(getindex.(triplets, [1 2 3])))

    violations = fit!(loss, triplets, X)

    return X
end

# end