# Stuff (mostly adjoints) to make things play nice with Zygote

# Needed as CompGraph creates a dict in a way which Zygote can not differentiate
Flux.Zygote.@adjoint! function getfield(p::Pair, i)
    getfield(p, i), Δ -> nothing
end
Flux.Zygote.@nograd Dict
Flux.Zygote.@adjoint! function convert(t::Type{T}, x::AbstractDict) where T<:AbstractDict
    convert(t, x), Δ -> nothing
end

Flux.Zygote.@nograd mutate
Flux.Zygote.@adjoint! function dispatch!(lm::LazyMutable, m::ResetLazyMutable, x...)
    dispatch!(lm, m, x...), Δ -> nothing
end

function ∇getdictkey(d::AbstractDict, k, ctx, Δ)
    grad = Flux.Zygote.grad_mut(ctx, d)
    grad[k] = Flux.Zygote.accum(get(grad, k, nothing), Δ)
    return (nothing, grad, nothing)
end

Flux.Zygote.@adjoint! function get!(f::Function, d::AbstractDict, k)
    # Will be replaced if ∇f is called
    back = Δ -> ∇getdictkey(d, k, __context__, Δ)

    function ∇f()
        res,fback = Flux.Zygote.pullback(__context__,f)
        back = function(Δ)
                Δd = get(Flux.Zygote.grad_mut(__context__, d), k, nothing)
                delete!(Flux.Zygote.grad_mut(__context__, d), k)
                fback(Δ) # Always return empty tuple due to no arg?
                return (nothing, Δd, nothing)
            end
        return res
    end
    return get!(∇f, d, k), back
end
