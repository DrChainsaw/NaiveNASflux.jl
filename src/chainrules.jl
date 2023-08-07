# Stuff to make things play nice with AD (typically Zygote)

# Not exported, but I ended up using this in NaiveGAflux and don't want to update it now...
# Consider it deprecated
const nograd = ChainRulesCore.ignore_derivatives

ChainRulesCore.@non_differentiable mutate(args...)

# Temp workaround for https://github.com/FluxML/Zygote.jl/issues/1111
# Whole function can be deleted if/when issue is resolved
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, m::MutableLayer, args...)
    res, back = rrule_via_ad(config, m.layer, args...)
    function MutableLayer_back(Δ)
        δs = back(Δ)
        Tangent{MutableLayer}(layer=δs[1]), δs[2:end]...
    end
    return res, MutableLayer_back
end

# Not just a workaround!
# We do forcemutation so that we don't end up trying to differentiate through it
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, m::LazyMutable, args...)
    forcemutation(m)
    res, back = rrule_via_ad(config, m.mutable, args...)
    function LazyMutable_back(Δ)
        δs = back(Δ)
        Tangent{LazyMutable}(mutable=δs[1]), δs[2:end]...
    end
    return res, LazyMutable_back
end

