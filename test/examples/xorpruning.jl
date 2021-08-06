md"""
# Model Pruning Example

While NaiveNASflux does not come with any built in search policies, it is still possible to do some cool stuff with it. 
Below is a very simple example of parameter pruning of a model trained on the `xor` function.

First we need some boilerplate to create the model and do the training:

"""
@testset "Pruning xor example" begin #hide
using NaiveNASflux, Flux, Test
using Flux: train!, mse
import Random
Random.seed!(666)
niters = 50

# To cut down on the verbosity, start by making a  helper function for creating a Dense layer as a graph vertex.
# The keyword argument `layerfun=`[`ActivationContribution`](@ref) will wrap the layer and compute an activity
# based neuron utility metric for it while the model trains.
densevertex(in, outsize, act) = fluxvertex(Dense(nout(in),outsize, act), in, layerfun=ActivationContribution)

# Ok, lets create the model and train it. We overparameterize quite heavily to avoid sporadic test failures :)
invertex = denseinputvertex("input", 2)
layer1 = densevertex(invertex, 32, relu)
layer2 = densevertex(layer1, 1, sigmoid)
original = CompGraph(invertex, layer2)

## Training params, nothing to see here
opt = ADAM(0.1)
loss(g) = (x, y) -> mse(g(x), y)

## Training data: xor truth table: y = xor(x)
x = Float32[0 0 1 1;
            0 1 0 1]
y = Float32[0 1 1 0]

## Train the model
train!(loss(original), params(original), Iterators.repeated((x,y), niters), opt)
@test loss(original)(x, y) < 0.001

# With that out of the way, lets try three different ways to prune the hidden layer (vertex nr 2 in the graph). 
# To make examples easier to compare, lets decide up front that we want to remove half of the hidden layer neurons 
# and try out three different ways of how to select which ones to remove.  
nprune = 16

# Prune the neurons with lowest utility according to the metric in ActivationContribution.
# This is the default if no utility function is provided.
pruned_least = deepcopy(original)
Δnout!(pruned_least[2] => -nprune)

# Prune the neurons with hiest utility according to the metric in ActivationContribution.
# This is obviously not a good idea if you want to preserve the accuracy.
pruned_most = deepcopy(original)
Δnout!(pruned_most[2] => -nprune) do v
    vals = NaiveNASlib.defaultutility(v)
    return 2*sum(vals) .- vals # Ensure all values are still > 0, even for last vertex
end

# Prune randomly selected neurons by giving random utility.
pruned_random = deepcopy(original)
Δnout!(v -> rand(nout(v)), pruned_random[2] => -nprune)

# Free lunch anyone?
@test   loss(pruned_most)(x, y)   > 
        loss(pruned_random)(x, y) > 
        loss(pruned_least)(x, y)  >= 
        loss(original)(x, y)

# The metric calculated by ActivationContribution is actually quite good in this case.
@test loss(pruned_least)(x, y) ≈ loss(original)(x, y) atol = 1e-5
end #hide




