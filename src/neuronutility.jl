"""
    ActivationContribution{L,C,M} <: AbstractMutableComp
    ActivationContribution(l)
    ActivationContribution(l, method)

Calculate neuron utility based on activations and gradients using `method`. 

Designed to be used as `layerfun` argument to [`fluxvertex`](@ref).

Can be a performance bottleneck in cases with large activations. Use [`NeuronUtilityEvery`](@ref) to mitigate.

Default `method` is described in <https://arxiv.org/abs/1611.06440>.

Short summary is that the first order taylor approximation of the optimization problem: "which neurons shall I remove to minimize impact on the loss function?" 
boils down to: "the ones which minimize `abs(gradient * activation)`" (assuming parameter independence).

"""
struct ActivationContribution{L,C,M} <: AbstractMutableComp
    layer::L
    contribution::C
    method::M
end
# We use eps(Float32) here because we don't want parameters from new layers to have:
# 1) higher utility than parameters from existing layers
# 2) zero utility since that will often make the optimizer remove them completely
# eps(Float32) is typically smaller than the optimizer tolerance, but NaiveNASlib tries to rescale
ActivationContribution(l::AbstractMutableComp, method = Ewma(0.05f0)) = ActivationContribution(l, fill(eps(Float32), nout(l)), method)
ActivationContribution(l, method = Ewma(0.05f0)) = ActivationContribution(l, Float32[], method)

@layer :expand ActivationContribution

wrapped(m::ActivationContribution) = m.layer

# We do train contribution in some sense, but we don't want Flux to do it
# We could create a "fake" gradient in the rrule and let the optimizer rule update it for us 
# (rather than using our own Ewma), but it is probably not desirable to mix the model parameter update
# strategy with the activation contribution strategy.
Flux.trainable(m::ActivationContribution) = (;layer = m.layer)

# Just passthrough when not taking gradients. 
(m::ActivationContribution)(x...) = wrapped(m)(x...)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, m::T, x...) where T <:ActivationContribution
    act, back = rrule_via_ad(config, wrapped(m), x...)

    function ActivationContribution_back(Δ)
        if length(m.contribution) === 0
            newcontribution = m.method(missing, act, Δ)
            resize!(m.contribution, length(newcontribution))
            copyto!(m.contribution, newcontribution)
        else
            copyto!(m.contribution, m.method(m.contribution, act, Δ))
        end

        δs = back(Δ)
        Tangent{T}(layer=δs[1]), δs[2:end]...
    end
    return act, ActivationContribution_back
end

actdim(nd::Integer) = nd - 1

function NaiveNASlib.Δsize!(m::ActivationContribution, inputs::AbstractVector, outputs::AbstractVector; kwargs...)
    if length(m.contribution) !== 0
        # This tends to happen when we are measuring contribution for a concatenation and we have added an extra input edge
        # TODO: Try to find another fix, perhaps we need to ensure that nout(v) if v wraps an ActivationContribution always return
        # the length of m.contribution
        outputs[outputs .> length(m.contribution)] .= -1 
        newcontribution = select(m.contribution, 1 => outputs; newfun = (args...) -> eps(eltype(m.contribution)))
        resize!(m.contribution, length(newcontribution))
        copyto!(m.contribution, newcontribution)
    #else 
    #   no need to select anything
    end
    NaiveNASlib.Δsize!(wrapped(m), inputs, outputs; kwargs...)
end


"""
    l2_squeeze(x, dimkeep)

Return l2 norm of `x` along all dimensions except `dimkeep` as a 1D array (singleton dimensions are removed).
"""
function l2_squeeze(x, dimskeep=1:ndims(x))
    dims = filter(i -> i ∉ dimskeep, 1:ndims(x))
    return sqrt.(dropdims(sum(x -> x^2, x, dims=dims), dims=Tuple(dims)))
end
l2_squeeze(z::Number, args...) = z

"""
    mean_squeeze(f, x, dimkeep)

Return mean value of `f.(x)` along all dimensions except `dimkeep` as a 1D array (singleton dimensions are removed).
"""
function mean_squeeze(f, x, dimskeep=1:ndims(x))
    dims = filter(i -> i ∉ dimskeep, 1:ndims(x))
    return dropdims(mean(f, x, dims=dims), dims=Tuple(dims))
end


# To peel the onion...
neuronutility(v::AbstractVertex) = neuronutility(base(v))
neuronutility(v::InputSizeVertex) = ones(nout(v))
neuronutility(v::CompVertex) = neuronutility(v.computation)
neuronutility(m::AbstractMutableComp) = neuronutility(wrapped(m))
function neuronutility(lm::LazyMutable)
   forcemutation(lm)
   neuronutility(wrapped(lm)) 
end
# Return missing to maintain API since previous versions used missing as sentinel value instead of empty vector
neuronutility(m::ActivationContribution) = isempty(m.contribution) ? missing : m.contribution
neuronutility(l) = neuronutility(layertype(l), l)

# Default: mean of abs of weights + bias. Not a very good metric, but should be better than random
# Maybe do something about state in recurrent layers as well, but CBA to do it right now
neuronutility(::FluxParLayer, l) = l2_squeeze(weights(l), outdim(l)) .+ l2_squeeze(bias(l))
function neuronutility(::FluxConvolutional{N}, l) where N
    ngroups(l) == 1 && return l2_squeeze(weights(l), outdim(l)) .+ l2_squeeze(bias(l))

    kernelsize = size(weights(l))[1:N]
    weightgroups = reshape(weights(l), kernelsize..., nout(l) ÷ ngroups(l), nin(l)[])

    wm = l2_squeeze(weightgroups, indim(l))
    bm = l2_squeeze(bias(l))

    (length(wm) == 1 || length(wm) == length(bm)) && return wm .+ bm
    # use this to get insight on whether to repeat inner or outer:
    # cc = DepthwiseConv(reshape(Float32[1 1 1 1;2 2 2 2], 1, 1, 4, 2), Float32[0,0,0,0,1,1,1,1])
    # cc(fill(10f0, (1,1,4,1)))
    return repeat(wm, length(bm) ÷ length(wm)) .+ bm
end

neuronutility(::FluxParNorm, l) = l.affine ? l2_squeeze(l.γ) .+ l2_squeeze(l.β) : missing 

# Not possible to do anything since we don't know the size. Implementors can however use this to fallback to other ways if this is not an error
neuronutility(lt, l) = missing

neuronutility_safe(v) = neuronutility_safe(trait(v), v) 
neuronutility_safe(t::DecoratingTrait, v) = neuronutility_safe(base(t), v)
neuronutility_safe(::Immutable, v) = 1
neuronutility_safe(::MutationSizeTrait, v) = clean_values(cpu(neuronutility(v)))
neuronutility_safe(m::AbstractMutableComp) = clean_values(cpu(neuronutility(m)))

clean_values(::Missing) = 1
clean_values(a::AbstractArray) = length(a) === 0 ? 1 : replace(a, NaN => -100, Inf => -100, -Inf => -100)

"""
    neuronutilitytaylor(currval, act, grad)

Calculate contribution of activations towards loss according to https://arxiv.org/abs/1611.06440.

Short summary is that the first order taylor approximation of the optimization problem: "which neurons shall I remove to minimize impact on the loss function?" 
boils down to: "the ones which minimize abs(gradient * activation)" (assuming parameter independence).
"""
neuronutilitytaylor(currval, act, grad) = mean_squeeze(abs, (mean_squeeze(identity, act .* grad, (actdim(ndims(act)), ndims(act)))), 1)
# Kinda wished they had branded this better as 'taylor' can mean many things. 

"""
    Ewma{R<:Real, M}
    Ewma(α::Real, method)

Exponential moving average of neuron utility calculated by `method`.

Parameter `α` acts as a forgetting factor, i.e larger values means faster convergence but more noisy estimate.
"""
struct Ewma{R<:Real, M}
    α::R
    method::M
    function Ewma(α::R, method::M) where {R,M}
        0 <= α <= 1 || error("α must be between 0 and 1, was $α")
        new{R,M}(α, method)
    end
end
Ewma(α) = Ewma(α, neuronutilitytaylor)

Functors.@leaf Ewma

(m::Ewma)(currval, act, grad) = agg(m, currval, m.method(currval, act, grad))

function agg(m::Ewma, x, y) 
    α = convert(float(eltype(x)), m.α)
    α .* x .+ (1 - α) .* y
end
agg(m::Ewma, ::Missing, y) = y


"""
    NeuronUtilityEvery{N,T}
    NeuronUtilityEvery(n::Int, method::T)

Calculate neuron utility using `method` every `n`:th call.

Useful to reduce runtime overhead.
"""
mutable struct NeuronUtilityEvery{N,T}
    cnt::Int
    method::T
    NeuronUtilityEvery(N::Int, method::T) where T = new{N, T}(0, method)
end
NeuronUtilityEvery(n::Int) = NeuronUtilityEvery(n, Ewma(0.05))

Functors.@leaf NeuronUtilityEvery # Or else bad things will happen (see inner constructor)!

function (m::NeuronUtilityEvery{N})(currval, act, grad) where N
    ret = m.cnt % N == 0 ? m.method(currval, act, grad) : currval
    m.cnt += 1
    return ret
end
