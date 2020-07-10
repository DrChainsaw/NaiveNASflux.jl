# NaiveNASflux

[![Build Status](https://travis-ci.org/DrChainsaw/NaiveNASflux.jl.svg?branch=master)](https://travis-ci.org/DrChainsaw/NaiveNASflux.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASflux.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASflux-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASflux.jl)

NaiveNASflux uses [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl) to enable mutation operations of arbitrary [Flux](https://github.com/FluxML/Flux.jl) computation graphs. It is designed with Neural Architecture Search (NAS) in mind, but can be used for any purpose where doing changes to a model is desired.

The following operations are supported:
* Change the input/output size of layers
* Parameter pruning, including methods to rank neurons
* Add layers to the model
* Remove layers from the model
* Add inputs to layers
* Remove inputs to layers

Note that NaiveNASflux does not have any functionality to search for a model architecture. Check out [NaiveGAflux](https://github.com/DrChainsaw/NaiveGAflux.jl) for a simple proof of concept.  

## Basic Usage

```julia
]add NaiveNASflux
```

Check out the basic usage of [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl) for less verbose examples.

Here is a quick rundown of some common operations:

```julia
# Input type: 3 channels 2D image
invertex = inputvertex("in", 3, FluxConv{2}())

# Mutable layers
conv = mutable(Conv((3,3), 3 => 5, pad=(1,1)), invertex)
batchnorm = mutable(BatchNorm(nout(conv), relu), conv)

# Explore graph
@test inputs(conv) == [invertex]
@test outputs(conv) == [batchnorm]

@test nin(conv) == [3]
@test nout(conv) == 5

@test layertype(conv) isa FluxConv{2}
@test layertype(batchnorm) isa FluxBatchNorm

# naming vertices is a good idea for debugging and logging purposes
namedconv = mutable("namedconv", Conv((5,5), 3=>7, pad=(2,2)), invertex)

@test name(namedconv) == "namedconv"

# Concatenate activations
conc = concat("conc", namedconv, batchnorm)

@test nout(conc) == nout(namedconv) + nout(batchnorm)

residualconv = mutable("residualconv", Conv((3,3), nout(conc) => nout(conc), pad=(1,1)), conc)

# Elementwise addition. '>>' operation can be used to add metadata, such as a name in this case
add = "add" >> conc + residualconv

@test name(add) == "add"
@test inputs(add) == [conc, residualconv]

# Computation graph for evaluation
graph = CompGraph(invertex, add)

# Can be evaluated just like any function
x = ones(Float32, 7, 7, nout(invertex), 2)
@test size(graph(x)) == (7, 7, nout(add), 2) == (7 ,7, 12 ,2)

# Graphs can be copied
graphcopy = copy(graph)

# Mutate number of neurons
@test nout(add) == nout(residualconv) == nout(conv) + nout(namedconv) == 12
Δnout(add, -3)
@test nout(add) == nout(residualconv) == nout(conv) + nout(namedconv) == 9

# Remove layer
@test nv(graph) == 7
remove!(batchnorm)
@test nv(graph) == 6

# Add layer
insert!(residualconv, v -> mutable(BatchNorm(nout(v), relu), v))
@test nv(graph) == 7


# Change kernel size (and supply new padding)
let Δsize = (-2, -2), pad = (1,1)
    mutate_weights(namedconv, KernelSizeAligned(Δsize, pad))
end

# Apply all mutations
apply_mutation(graph)

# Note: Parameters not changed yet...
size(NaiveNASflux.weights(layer(namedconv))) == (5, 5, 3, 7)

@test size(graph(x)) == (7, 7, nout(add), 2) == (7, 7, 9, 2)

# ... because mutations are lazy by default so that no new layers are created until the graph is evaluated
size(NaiveNASflux.weights(layer(namedconv))) == (3, 3, 3, 7)

# Btw, the copy we made above is of course unaffected
@test size(graphcopy(x)) == (7, 7, 12, 2)
```

While NaiveNASflux does not come with any built in search policies, it is still possible to do some cool stuff with it. Below is a very simple example of parameter pruning of a model trained on the `xor` function.

First we need some boilerplate to create the model and do the training:

```julia
using NaiveNASflux, Test
import Flux: train!, mse
import Random
Random.seed!(666)
niters = 50

# First lets create a simple model
# layerfun=ActivationContribution will wrap the layer and compute a pruning metric for it while the model trains
mdense(in, outsize, act) = mutable(Dense(nout(in),outsize, act), in, layerfun=ActivationContribution)

invertex = inputvertex("input", 2, FluxDense())
layer1 = mdense(invertex, 32, relu)
layer2 = mdense(layer1, 1, sigmoid)
original = CompGraph(invertex, layer2)

# Training params, nothing to see here
opt = ADAM(0.1)
loss(g) = (x, y) -> mse(g(x), y)

# Training data: xor truth table: y = xor(x)
x = Float32[0 0 1 1;
            0 1 0 1]
y = Float32[0 1 1 0]

# Train the model
for iter in 1:niters
    train!(loss(original), params(original), [(x,y)], opt)
end
@test loss(original)(x, y) < 0.001
```

With that out of the way, lets try three different ways to prune the hidden layer:

```julia
nprune = 16

# Prune randomly selected neurons
pruned_random = copy(original)
Δnin(pruned_random.outputs[], -nprune)
Δoutputs(pruned_random.outputs[], v -> rand(nout_org(v)))
apply_mutation(pruned_random)

# Prune the least valuable neurons according to the metric in ActivationContribution
pruned_least = copy(original)
Δnin(pruned_least.outputs[], -nprune)
Δoutputs(pruned_least.outputs[], neuron_value)
apply_mutation(pruned_least)

# Prune the most valuable neurons according to the metric in ActivationContribution
pruned_most = copy(original)
Δnin(pruned_most.outputs[], -nprune)
Δoutputs(pruned_most.outputs[], v -> -neuron_value(v))
apply_mutation(pruned_most)

# Can I have my free lunch now please?!
@test loss(pruned_most)(x, y) > loss(pruned_random)(x, y) > loss(pruned_least)(x, y) > loss(original)(x, y)

# The metric calculated by ActivationContribution is actually quite good (in this case).
@test loss(pruned_least)(x, y) ≈ loss(original)(x, y) atol = 1e-5
```

Another toy example where the model has too few layers to efficiently fit the data.

Create model and train it just to have something to mutate:
```julia
using NaiveNASflux, Test
import Flux.Losses
import Flux.Losses: logitbinarycrossentropy
import Flux: train!, glorot_uniform
using Statistics
import Random
Random.seed!(666)
niters = 20

# Layers used in this example
mconv(in, outsize, act; init=glorot_uniform) = mutable(Conv((3,3),nout(in)=>outsize, act, pad=(1,1), init=init), in)
mavgpool(in, h, w) = mutable(MeanPool((h, w)), in)
mdense(in, outsize, act) = mutable(Dense(nout(in),outsize, act), in)

# Size of the input
height = 4
width = 4

function model(nconv)
    invertex = inputvertex("in", 1, FluxConv{2}())
    l = invertex
    for i in 1:nconv
        l = mconv(l, 16, relu)
    end
    l = mavgpool(l, height, width)
    l = invariantvertex(Flux.flatten, l)
    l = mdense(l, 2, identity)
    return CompGraph(invertex, l)
end
original = model(1)

# Training params, nothing to see here
opt_org = ADAM(0.01)
loss(g) = (x, y) -> logitbinarycrossentropy(g(x), y, agg=mean)

# Training data: 2D xor(-ish)
# Class 1 are matrices A where A[n,m] ⊻ A[n+h÷2, m]) ⩓ A[n,m] ⊻ A[n, m+w÷2] ∀ n<h÷2, m<w÷2 is true, e.g. [1 0; 0, 1]
function xy1(h,w)
    q1_3 = rand(Bool,h÷2,w÷2)
    q2_4 = .!q1_3
    return (Float32.(vcat(hcat(q2_4, q1_3), hcat(q1_3, q2_4))), Float32[0, 1])
end
xy0(h,w) = (rand(Float32[0,1], h,w), Float32[1, 0]) #Joke's on me when this generates false negatives :)
# Generate 50% class 1 and 50% class 0 examples in one batch
catbatch((x1,y1)::Tuple, (x2,y2)::Tuple) = (cat(x1, x2, dims=4), hcat(y1,y2))
batch(h,w,batchsize) = mapfoldl(i -> i==0 ? xy0(h,w) : xy1(h,w), catbatch, (1:batchsize) .% 2)

x_test, y_test = batch(height, width, 1024)
startloss = loss(original)(x_test, y_test)

# Train the model
for iter in 1:niters
    train!(loss(original), params(original), [batch(height, width, 64)], opt_org)
end
# That didn't work so well...
@test loss(original)(x_test, y_test) ≈ startloss atol=1e-1

# Lets try three things:
# 1. Just train the same model some more
# 2. Add two more conv-layers to the already trained model
# 3. Create a new model with three conv layers from scratch

# Disclaimer: This experiment is intended to show usage of this library.
# It is not meant to give evidence that method 2 is the better option.
# Hyperparameters are tuned to strongly favor 2 in order to avoid sporadic failures

# Add two layers after the conv layer
add_layers = copy(original)
function add2conv(in)
    l = mconv(in, nout(in), relu, init=idmapping)
    return mconv(l, nout(in), relu, init=idmapping)
end
insert!(vertices(add_layers)[2], add2conv)

# New layers are initialized to identity mapping weights
# We basically have the same model as before, just with more potential
# Not guaranteed to be a good idea as it relies on existing layers to provide gradient diversity
@test add_layers(x_test) == original(x_test)

# Create a new model with three conv layers
new_model = model(3)

opt_add = deepcopy(opt_org)
opt_new = ADAM(0.01)

# Lets try again
for iter in 1:niters
    b = batch(height, width, 64)
    train!(loss(original), params(original), [b], opt_org)
    train!(loss(add_layers), params(add_layers), [b], opt_add)
    train!(loss(new_model), params(new_model), [b], opt_new)
end

@test loss(add_layers)(x_test,y_test) < loss(new_model)(x_test,y_test) < loss(original)(x_test,y_test)
```

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
