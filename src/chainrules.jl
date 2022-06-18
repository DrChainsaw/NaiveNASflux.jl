# Stuff (mostly adjoints) to make things play nice with Zygote

nograd(f) = f()
ChainRulesCore.@non_differentiable nograd(f)

ChainRulesCore.@non_differentiable mutate(args...)

function ChainRulesCore.rrule(::typeof(dispatch!), lm::LazyMutable, m::ResetLazyMutable, x...)
    dispatch!_back(Δ) = ntuple(_ -> NoTangent(), length(x) + 2)
    dispatch!(lm, m, x...), dispatch!_back
end

# Temp workaround for https://github.com/FluxML/Zygote.jl/issues/1111
# Whole function can be deleted if/when issue is resolved
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, m::MutableLayer, args...)
    res, back = rrule_via_ad(config, m.layer, args...)
    function MutableLayer_back(Δ)
        δlayer, δargs = back(Δ)
        Tangent{MutableLayer}(layer=δlayer), δargs
    end
    return res, MutableLayer_back
end

# Temp workaround for https://github.com/FluxML/Zygote.jl/issues/1111
# Whole function can be deleted if/when issue is resolved
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, m::LazyMutable, args...)
    forcemutation(m)
    res, back = rrule_via_ad(config, m.mutable, args...)
    function LazyMutable_back(Δ)
        δmutable, δargs = back(Δ)
        Tangent{LazyMutable}(mutable=δmutable), δargs
    end
    return res, LazyMutable_back
end

