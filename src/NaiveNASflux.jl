module NaiveNASflux

using Reexport
@reexport using NaiveNASlib
@reexport using Flux
import Flux.Zygote: hook
using Statistics
using Setfield
using LinearAlgebra
import InteractiveUtils: subtypes
import JuMP: @variable, @constraint

export FluxLayer, FluxParLayer, FluxDense, FluxRecurrent, FluxRnn, FluxGru, FluxLstm, FluxConvolutional, FluxConv, FluxConvTranspose, FluxDepthwiseConv, FluxCrossCor, FluxTransparentLayer, FluxParInvLayer, FluxDiagonal, FluxLayerNorm, FluxParNorm, FluxBatchNorm, FluxInstanceNorm, FluxGroupNorm, FluxNoParLayer

export mutable, concat, AbstractMutableComp, MutableLayer, LazyMutable, NoParams

export named, validated, logged

export ActivationContribution, neuron_value, Ewma

export indim, outdim, actdim, layer, layertype

export idmapping, idmapping_nowarn

export KernelSizeAligned, mutate_weights


include("types.jl")
include("util.jl")
include("select.jl")
include("mutable.jl")
include("vertex.jl")
include("pruning.jl")
include("weightinit.jl")


# Modified version of Flux.functor for mutable structs which are mutable mainly because they are intended to be wrapped in MutationVertices which in turn are not easy to create in the manner which Flux.functor is designed.
function mutable_makefunctor(m::Module, T, fs = functor_fields(T))
  @eval m begin
    Flux.functor(x::$T) = ($([:($f=x.$f) for f in fs]...),),
    # Instead of creating a new T, we set all fields in fs of x to y (since y is the fields returned in line above)
    function(y)
      $([:(x.$f = y[$i]) for (i, f) in enumerate(fs)]...)
      return x
    end
  end
end

function mutable_functorm(T, fs = nothing)
  fs == nothing || isexpr(fs, :tuple) || error("@functor T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(mutable_makefunctor(@__MODULE__, $(esc(T)), $(fs...)))
end

macro mutable_functor(args...)
  mutable_functorm(args...)
end

functor_fields(T) = fieldnames(T)

for subtype in subtypes(AbstractMutableComp)
  @mutable_functor subtype
end

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


end # module
