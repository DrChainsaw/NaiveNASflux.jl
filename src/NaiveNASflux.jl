module NaiveNASflux

using NaiveNASlib
using Flux

export mutable

export nin, nout, indim, outdim

include("types.jl")
include("util.jl")
include("mutable.jl")
include("vertex.jl")

end # module
