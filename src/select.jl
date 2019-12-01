
select(::Missing, elements_per_dim...; insval = 0) = missing

"""
    select(pars::AbstractArray{T,N}, elements_per_dim...; insval = 0) where {T, N}

Return a new `AbstractArray{T, N}` which has a subset of the elements of `pars`.

Which elements to select is determined by `elements_per_dim` which is a `Pair{Int, Vector{Int}}` mapping dimension (first memeber) to which elements to select in that dimension (second memeber).

For a single `dim=>elems` pair, the following holds: `selectdim(output, dim, i) == selectdim(pars, dim, elems[i])` if `elems[i]` is positive and `selectdim(output, dim, i) .== insval` if `elems[i]` is negative.

# Examples
```julia-repl
julia> using NaiveNASflux

julia> pars = reshape(1:3*5, 3,5)
3×5 reshape(::UnitRange{Int64}, 3, 5) with eltype Int64:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

 julia> NaiveNASflux.select(pars, 1 => [-1, 1,3,-1,2], 2=>[3, -1, 2], insval=-1)
 5×3 Array{Int64,2}:
  -1  -1  -1
   7  -1   4
   9  -1   6
  -1  -1  -1
   8  -1   5
```
"""
function select(pars::AbstractArray{T,N}, elements_per_dim...; insval = 0) where {T, N}
    psize = collect(size(pars))
    assign = repeat(Any[Colon()], N)
    access = repeat(Any[Colon()], N)

    for de in elements_per_dim
        dim = de.first
        elements = de.second

        indskeep = filter(ind -> ind > 0, elements)
        newmap = elements .> 0

        psize[dim] = length(newmap)
        assign[dim] = findall(newmap)
        access[dim] = indskeep
    end
    newpars = fill!(similar(pars, psize...), T(0)) .+ T(insval)
    newpars[assign...] = pars[access...]
    return newpars
end


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
