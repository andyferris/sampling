# This number type lets us deal with very big/small numbers in the sampling algorithm (only positive numbers here!)
immutable LogFloat64 <: Real
    x::Float64
    iszero::Bool
end
LogFloat64(in::Float64) = in == 0. ? LogFloat64(0.,true) : LogFloat64(log(in),false)

import Base.convert
convert(::Type{Float64},in::LogFloat64) = in.iszero ? 0. : exp(in.x)
convert(::Type{LogFloat64},in::Float64) = in == 0. ? LogFloat64(0.,true) : LogFloat64(log(in),false)

import Base.promote_rule
promote_rule(::Type{Float64},::Type{LogFloat64}) = LogFloat64

*(x::LogFloat64,y::LogFloat64) = x.iszero || y.iszero ? LogFloat64(0.,true) : LogFloat64(x.x+y.x,false)
.*(x::LogFloat64,y::LogFloat64) = x.iszero || y.iszero ? LogFloat64(0.,true) : LogFloat64(x.x+y.x,false)
/(x::LogFloat64,y::LogFloat64) = x.iszero ? LogFloat64(0.,true) : (y.iszero ? LogFloat64(Inf,false) : LogFloat64(x.x-y.x,false))
./(x::LogFloat64,y::LogFloat64) = x.iszero ? LogFloat64(0.,true) : (y.iszero ? LogFloat64(Inf,false) : LogFloat64(x.x-y.x,false))
+(x::LogFloat64,y::LogFloat64) = x.iszero ? y : (y.iszero ? x : (x.x > y.x ? LogFloat64(x.x + log(1.0 + exp(y.x-x.x)),false) : LogFloat64(y.x + log(1.0 + exp(x.x-y.x)),false)))

import Base.zero
import Base.one
zero(::Type{LogFloat64}) = LogFloat64(0.,true)
zero(::LogFloat64) = LogFloat64(0.,true)
one(::Type{LogFloat64}) = LogFloat64(0.,false)
one(::LogFloat64) = LogFloat64(0.,false)


function choose_importance_sampling(W::Vector{Float64},k::Int64)
    n = length(W)
    if k >= n
       return([1:n],ones(n),ones(n))
    end

    # Make it better balanced to avoid overflows. In this way the most likely combination has weight 1.
    # I have observed cases where the most likely combination suffers an underflow
    #W_sorted = sort(W,rev=true)
    #W /= prod(W_sorted[1:k].^(1.0/k))
    # NOTE: the above approach might be useful for MATLAB but is unnecessary with the LogFloat64 type

    W = convert(Vector{LogFloat64},W)

    ps = zeros(Float64,n)

    # Create Ls
    Ls = Array(Array{LogFloat64,1},n+1)
    Rs = Array(Array{LogFloat64,1},n+1)

    Ls[1] = zeros(LogFloat64,k+1)
    Ls[1][1] = one(LogFloat64)
    for i = 2:n+1
        Ls[i] = Ls[i-1] + [zero(LogFloat64),W[i-1]*Ls[i-1][1:end-1]]
    end

    # Create Rs and extract marginal probabilities
    Rs[n+1] = zeros(LogFloat64,k+1)
    Rs[n+1][k+1] = one(LogFloat64)
    for i = n:-1:1
        p1 = sum(Ls[i] .* Rs[i+1])
        p2 = W[i] * sum(Ls[i][1:end-1] .* Rs[i+1][2:end])

        ps[i] = convert(Float64,p2/(p1+p2))

        Rs[i] = Rs[i+1] + [W[i]*Rs[i+1][2:end],zero(LogFloat64)]
    end

    # Sweep again from left, choosing samples withs sequential perfect sampling.
    indices = zeros(Int64,k)
    weights = zeros(Float64,n)

    ii = 1
    for i = 1:n
        p1 = sum(Ls[i] .* Rs[i+1])
        p2 = W[i] * sum(Ls[i][1:end-1] .* Rs[i+1][2:end])
        p = convert(Float64,p2/(p1+p2))

        if rand() < p
            indices[ii] = i
            ii += 1

            weights[i] = 1.0/ps[i]
            if ii > k
                break
            end

            Ls[i+1] = [zero(LogFloat64),W[i]*Ls[i][1:end-1]] / p
        else
            weights[i] = 0.0

            Ls[i+1] = Ls[i] / (1.0-p)
        end
    end

    if ii <= k # This event should occur with probabilty zero.
        warning("Sampling error - generated less than requested samples")
    end

    return (indices, ps, weights)
end
