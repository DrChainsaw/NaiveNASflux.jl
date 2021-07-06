"""
    ActivationContribution{L,M} <: AbstractMutableComp
    ActivationContribution(l)
    ActivationContribution(l, method)

Calculate neuron value based on activations and gradients using `method` (default EWMA of [`NeuronValueTaylor`](@ref)).

High value of `contribution` means that the neuron of that index has a high contribution to the loss score.

Can be a performance bottleneck in cases with large activations. Use [`NeuronValueEvery`](@ref) to mitigate.
"""
mutable struct ActivationContribution{L,M} <: AbstractMutableComp
    layer::L
    contribution # Type of activation not known yet :( Also leave some room for experimenting with things like storing the metric on the GPU
    method::M
end
ActivationContribution(l::AbstractMutableComp, method = Ewma(0.05)) = ActivationContribution(l, zeros(Float32, nout(l)), method)
ActivationContribution(l, method = Ewma(0.05)) = ActivationContribution(l, neuron_value_safe(l), method)

layer(m::ActivationContribution) = layer(m.layer)
layertype(m::ActivationContribution) = layertype(m.layer)
wrapped(m::ActivationContribution) = m.layer
NaiveNASlib.minΔninfactor(m::ActivationContribution) = minΔninfactor(wrapped(m))
NaiveNASlib.minΔnoutfactor(m::ActivationContribution) = minΔnoutfactor(wrapped(m))
functor_fields(::Type{ActivationContribution}) = (:layer,)

function(m::ActivationContribution)(x...)
    act = wrapped(m)(x...)

    return hook(act) do grad
        grad === nothing && return grad
        m.contribution = m.method(m.contribution, act, grad)
        return grad
    end
end

actdim(nd::Integer) = nd - 1

function NaiveNASlib.Δsize!(m::ActivationContribution, inputs::AbstractVector, outputs::AbstractVector; kwargs...)
    m.contribution = select(m.contribution, 1 => outputs; newfun = (args...) -> 0)
    NaiveNASlib.Δsize!(wrapped(m), inputs, outputs; kwargs...)
end

"""
    mean_squeeze(x, dimkeep)

Return mean value of `x` along all dimensions except `dimkeep` as a 1D array (singleton dimensions are removed).
"""
function mean_squeeze(f, x, dimskeep)
    dims = filter(i -> i ∉ dimskeep, 1:ndims(x))
    return dropdims(mean(f, x, dims=dims), dims=Tuple(dims))
end

# To peel the onion...
neuron_value(v::AbstractVertex) = neuron_value(base(v))
neuron_value(v::InputSizeVertex) = ones(nout(v))
neuron_value(v::CompVertex) = neuron_value(v.computation)
neuron_value(m::AbstractMutableComp) = neuron_value(wrapped(m))
function neuron_value(lm::LazyMutable)
   forcemutation(lm)
   neuron_value(wrapped(lm)) 
end
neuron_value(m::ActivationContribution) = m.contribution
neuron_value(l) = neuron_value(layertype(l), l)

# Default: mean of abs of weights + bias. Not a very good metric, but should be better than random
# Maybe do something about state in recurrent layers as well, but CBA to do it right now
neuron_value(::FluxParLayer, l) = mean_squeeze(abs, weights(l), outdim(l)) .+ bcabsz(bias(l))
function neuron_value(::FluxDepthwiseConv, l)
    wm = mean_squeeze(abs, weights(l), outdim(l))
    bm = bcabsz(bias(l))

    (length(wm) == 1 || length(wm) == length(bm)) && return wm .+ bm
    # use this to get insight on whether to repeat inner or outer:
    # cc = DepthwiseConv(reshape([1 1 1 1;2 2 2 2], 1, 1, 2, 4), [0,0,0,0,1,1,1,1])
    # cc(fill(10, (1,1,4,1)))
    return repeat(wm, length(bm) ÷ length(wm)) .+ bm
end
bcabsz(x) = abs.(x)
bcabsz(z::Flux.Zeros) = z

# Not possible to do anything since we don't know the size. Implementors can however use this to fallback to other ways if this is not an error
neuron_value(lt, l) = missing

neuron_value_safe(v) = neuron_value_safe(trait(v), v) 
neuron_value_safe(t::DecoratingTrait, v) = neuron_value_safe(base(t), v)
neuron_value_safe(::Immutable, v) = ones(nout(v))
neuron_value_safe(::MutationSizeTrait, v) = clean_values(cpu(neuron_value(v)),v)
neuron_value_safe(m::AbstractMutableComp) = clean_values(cpu(neuron_value(m)), m)

clean_values(::Missing, v) = ones(nout(v))
clean_values(a::AbstractArray, v) = replace(a, NaN => -100, Inf => -100, -Inf => -100)

"""
    neuronvaluetaylor(currval, act, grad)

Calculate contribution of activations towards loss according to https://arxiv.org/abs/1611.06440.

Short summary is that the first order taylor approximation of the optimization problem: "which neurons shall I remove to minimize impact on the loss function?" boils down to: "the ones which minimize abs(gradient * activation)" (assuming parameter independence).
"""
neuronvaluetaylor(currval, act, grad) = mean_squeeze(abs, (mean_squeeze(identity, act .* grad, (actdim(ndims(act)), ndims(act)))), 1)


"""
    Ewma{R<:Real, M}
    Ewma(α::Real, method)

Exponential moving average of neuron value calculated by `method`.

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
Ewma(α) = Ewma(α, neuronvaluetaylor)

(m::Ewma)(currval, act, grad) = agg(m, currval, m.method(currval, act, grad))

# Basically for backwards compatibility even though method is not exported
agg(m::Ewma, x, y) = m.α .* cpu(x) .+ (1 - m.α) .* cpu(y)
agg(m::Ewma, ::Missing, y) = cpu(y)


"""
    NeuronValueEvery{N,T}
    NeuronValueEvery(n::Int, method::T)

Calculate neuron value using `method` every `n`:th call.

Useful to reduce runtime overhead.
"""
mutable struct NeuronValueEvery{N,T}
    cnt::Int
    method::T
    NeuronValueEvery(N::Int, method::T) where T = new{N, T}(0, method)
end
NeuronValueEvery(n::Int) = NeuronValueEvery(n, Ewma(0.05))

function (m::NeuronValueEvery{N})(currval, act, grad) where N
    ret = m.cnt % N == 0 ? m.method(currval, act, grad) : currval
    m.cnt += 1
    return ret
end
