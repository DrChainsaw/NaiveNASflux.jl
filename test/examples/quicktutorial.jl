md"""
# Quick Tutorial

Check out the basic usage of [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl) for less verbose examples.

Here is a quick rundown of some common operations.

"""
@testset "Quick tutorial" begin #src
using NaiveNASflux, Flux, Test

# Create an input vertex which tells its output vertices that they can expect 2D convolutional input (i.e 4D arrays).
invertex = conv2dinputvertex("in", 3)

# Vertex type for Flux-layers is automatically inferred through [`fluxvertex`](@ref).
conv = fluxvertex(Conv((3,3), 3 => 5, pad=(1,1)), invertex)
batchnorm = fluxvertex(BatchNorm(nout(conv), relu), conv)

# Explore graph.
@test inputs(conv) == [invertex]
@test outputs(conv) == [batchnorm]

@test nin(conv) == [3]
@test nout(conv) == 5

@test layer(conv) isa Flux.Conv
@test layer(batchnorm) isa Flux.BatchNorm

# Naming vertices is a good idea for debugging and logging purposes.
namedconv = fluxvertex("namedconv", Conv((5,5), 3=>7, pad=(2,2)), invertex)

@test name(namedconv) == "namedconv"

# Concatenate activations. Dimension is automatically inferred.
conc = concat("conc", namedconv, batchnorm)
@test nout(conc) == nout(namedconv) + nout(batchnorm)

# No problem to combine with convenience functions from NaiveNASlib.
residualconv = fluxvertex("residualconv", Conv((3,3), nout(conc) => nout(conc), pad=(1,1)), conc)
add = "add" >> conc + residualconv

@test name(add) == "add"
@test inputs(add) == [conc, residualconv]

# Computation graph for evaluation. It is basically a more generic version of `Flux.Chain`.
graph = CompGraph(invertex, add)

# Access the vertices of the graph.
@test vertices(graph) == [invertex, namedconv, conv, batchnorm, conc, residualconv, add]

# `CompGraph`s can be evaluated just like any function.
x = ones(Float32, 7, 7, nout(invertex), 2)
@test size(graph(x)) == (7, 7, nout(add), 2) == (7 ,7, 12 ,2)

# Mutate number of neurons.
@test nout(add) == nout(residualconv) == nout(conv) + nout(namedconv) == 12
Δnout!(add => -3)
@test nout(add) == nout(residualconv) == nout(conv) + nout(namedconv) == 9

# Remove layer.
@test nvertices(graph) == 7
remove!(batchnorm)
@test nvertices(graph) == 6

# Add layer.
insert!(residualconv, v -> fluxvertex(BatchNorm(nout(v), relu), v))
@test nvertices(graph) == 7

# Change kernel size (and supply new padding).
namedconv |> KernelSizeAligned(-2,-2; pad=SamePad())

# Note: Parameters not changed yet...
@test size(NaiveNASflux.weights(layer(namedconv))) == (5, 5, 3, 7)

# ... because mutations are lazy by default so that no new parameters are created until the graph is evaluated.
@test size(graph(x)) == (7, 7, nout(add), 2) == (7, 7, 9, 2)
@test size(NaiveNASflux.weights(layer(namedconv))) == (3, 3, 3, 4)
end #src




