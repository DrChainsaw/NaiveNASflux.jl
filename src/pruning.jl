"""
 Calculate contribution of activations towards loss according to https://arxiv.org/abs/1611.06440.

High value of `contribution` means that the neuron of that index has a high contribution to the loss score.

Short summary is that the first order taylor approximation of the optimization problem: "which neurons shall I remove to minimize impact on the loss function" boils down to: "the ones which minimize abs(gradient * activation)" (assuming parameter independence).
"""
mutable struct ActivationContribution <: AbstractMutableComp
    layer
    contribution
end
ActivationContribution(l::AbstractMutableComp) = ActivationContribution(l, zeros(Float32, nout(l)))
ActivationContribution(l) = ActivationContribution(l, missing)

layer(m::ActivationContribution) = layer(m.layer)
wrapped(m::ActivationContribution) = m.layer

function(m::ActivationContribution)(x)
    act = wrapped(m)(x)
    m.contribution = lazyinit(m.contribution, act)
    return hook(act) do grad
        m.contribution[1:end] += mean_squeeze(abs.(act .* grad).data, actdim(ndims(act)))
        return grad
    end
end
lazyinit(::Missing, arr::AbstractArray{T, N}) where {T,N}= fill!(similar(arr, size(arr, actdim(N))), T(0))
lazyinit(c, arr) = c

actdim(nd::Integer) = nd - 1

function NaiveNASlib.mutate_outputs(m::ActivationContribution, outputs::AbstractVector{<:Integer})
    m.contribution = select(m.contribution, 1 => outputs)
    mutate_outputs(wrapped(m), outputs)
end

"""
    mean_squeeze(x, dimkeep)

Return mean value of `x` along all dimensions except `dimkeep` as a 1D array (singleton dimensions are removed).
"""
function mean_squeeze(x, dimkeep)
    dims = filter(i -> i != dimkeep, 1:ndims(x))
    return dropdims(mean(x, dims=dims), dims=Tuple(dims))
end

# To peel the onion...
neuron_value(v::AbstractVertex) = neuron_value(base(v))
neuron_value(v::CompVertex) = neuron_value(v.computation)
neuron_value(m::AbstractMutableComp) = neuron_value(wrapped(m))
neuron_value(m::ActivationContribution) = m.contribution
neuron_value(l) = neuron_value(layertype(l), l)

# Default: highest mean of abs of weights + bias. Not a very good metric, but should be better than random
# Maybe do something about state in recurrent layers as well, but CBA to do it right now
neuron_value(t::FluxParLayer, l) = mean_squeeze(abs.(weights(l)), outdim(l)) + abs.(bias(l))
