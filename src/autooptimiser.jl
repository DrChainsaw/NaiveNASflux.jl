mutable struct AutoOptimiser{L} <: AbstractMutableComp
    layer::L
    optstate # We want to be able to change type of this for now
end
AutoOptimiser(o::Flux.Optimise.AbstractOptimiser) = l -> AutoOptimiser(l, Optimisers.setup(o, l))
AutoOptimiser(o::Optimisers.AbstractRule) = l -> AutoOptimiser(l, Optimisers.setup(o, l))
AutoOptimiser(l) = AutoOptimiser(l, nothing)
AutoOptimiser(m::NoParams) = m

@functor AutoOptimiser

wrapped(a::AutoOptimiser) = a.layer

# We should probably return empty/nothing here, but for now lets keep this so Flux.params works
# We anyways override _setup below...
Flux.trainable(a::AutoOptimiser) = (;layer=Flux.trainable(a.layer)) 

(a::AutoOptimiser)(args...) = wrapped(a)(args...)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, a::AutoOptimiser, args...)
    res, back = rrule_via_ad(config, a.layer, args...)
    function AutoOptimiser_back(Δ)
        δs = back(Δ)
        a.optstate = _updateparams_safe!(a.optstate, a.layer, δs[1])
        NoTangent(), δs[2:end]...
    end
    return res, AutoOptimiser_back
end

# This is a bit of a temp hack just to be able to try stuff out until the infrastructure in NaiveGAflux is updated
function Optimisers._setup(rule::Optimisers.AbstractRule, a::AutoOptimiser; kwargs...)
    a.optstate = Optimisers._setup(rule, a.layer; kwargs...)
    return nothing
end
_updateparams_safe!(::Nothing, args...) = throw(ArgumentError("AutoOptimiser without optimiser state invoked. Forgot to run setup?"))
function _updateparams_safe!(optstate, model, grads)
    deepany(grads) do g
        g isa Number || return false
        isnan(g) || isinf(g)      
    end && return optstate
    first(Optimisers.update!(optstate, model, grads))
end

deepany(f, x::Union{Tuple, NamedTuple}) = any(e -> deepany(f, e), x)
deepany(f, x::Nothing) = f(x)
deepany(f, x) = any(f, x)