
select(pars::AbstractArray{T,N}, elements_per_dim...; newfun = randoutzeroin) where {T, N} = NaiveNASlib.parselect(pars, elements_per_dim...; newfun)
select(::Missing, args...;kwargs...) = missing
select(::Flux.Zeros, args...;kwargs...) = Flux.Zeros()

struct WeightParam end
struct BiasParam end
struct RecurrentWeightParam end
struct RecurrentState end

"""
    neuroninsert(lt, partype)

Return a function which creates new parameters for layers of type `lt` to use for [`select`](@Ref).
"""
neuroninsert(t, partype) = randoutzeroin
neuroninsert(t, parname::Symbol) = neuroninsert(t, Val(parname))
neuroninsert(t, name::Val) = randoutzeroin

neuroninsert(lt::FluxParNorm, t::Val) = norminsert(lt, t)
norminsert(::FluxParNorm, ::Union{Val{:β},Val{:μ}}) = (args...) -> 0
norminsert(::FluxParNorm, ::Union{Val{:γ},Val{:σ²}}) = (args...) -> 1

# Coupling between input and output weights make it difficult to do anything else?
neuroninsert(::FluxDepthwiseConv, partype) = (args...) -> 0

randoutzeroin(T, d, s...) = _randoutzeroin(T,d,s)
_randoutzeroin(T, d, s) = 0
_randoutzeroin(T, d, s::NTuple{2, Int}) = d == indim(FluxDense()) ? 0 : randn(T, s) ./ prod(s)
_randoutzeroin(T, d, s::NTuple{N, Int}) where N = d == indim(FluxConv{N-2}()) ? 0 : randn(T, s) ./ prod(s)


"""
    KernelSizeAligned(Δsize)
    KernelSizeAligned(Δs::Integer...)

Strategy for changing kernel size of convolutional layers where filters remain phase aligned. In other words, the same element indices are removed/added for all filters and only 'outer' elements are dropped or added.
"""
struct KernelSizeAligned{T, P}
    Δsize::T
    pad::P
end
KernelSizeAligned(Δs::Integer...;pad = ntuple(i -> 0, length(Δs))) = KernelSizeAligned(Δs, pad)

(s::KernelSizeAligned)(l) = selectfilters(layertype(l), l, s)

otherpars(s::KernelSizeAligned, l) = paddingfor(layertype(l), s)
paddingfor(t, s) = ()
paddingfor(::FluxConvolutional, s) = (pad = s.pad,)

selectfilters(t, l, s) = ()
selectfilters(::FluxConvolutional, l, s) = selectfilters(s, weights(l))

function selectfilters(s::KernelSizeAligned, pars)
    csize = size(pars)
    mids = csize ./ 2 .+ 0.5
    byfun(dim) = i -> abs(mids[dim] - i) * 1 / (1 + mean(abs.(selectdim(pars, dim, i))))
    ps = Pair{Int, Vector{Int}}[]
    for (dim, Δsize) in enumerate(s.Δsize)
        if Δsize < 0
            els = sort(partialsort(1:csize[dim], 1:csize[dim]+Δsize, by=byfun(dim)))
            push!(ps, dim => els)
        elseif Δsize > 0
            els = -ones(Int, csize[dim] + Δsize)
            offs = Δsize ÷ 2
            els[offs+1:csize[dim]+offs] = 1:csize[dim]
            push!(ps, dim => els)
        end
    end
    return ps
end
