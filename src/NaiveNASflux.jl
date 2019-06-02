module NaiveNASflux

using NaiveNASlib
using Flux
using Setfield

export layertype

export mutable, MutableLayer, LazyMutable

export nin, nout, indim, outdim

include("types.jl")
include("util.jl")
include("mutable.jl")
include("vertex.jl")

end # module
