# NaiveNASflux

[![Build Status](https://travis-ci.com/DrChainsaw/NaiveNASflux.jl.svg?branch=master)](https://travis-ci.com/DrChainsaw/NaiveNASflux.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASflux.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASflux-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASflux.jl)

Work in progress!!

NaiveNASflux is a library for Neural Architecture Search (NAS) using [Flux](https://github.com/FluxML/Flux.jl) built on top of [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl). It offers powerful mutation operations of arbitrary computation graphs at your fingertips. No more wasting time on graph traversal and multi-dimensional array indexing!  

The following operations are supported:
* Change the input/output size of layers
* Parameter pruning, including methods to rank neurons
* Add layers to the model
* Remove layers from the model
* Add inputs to layers
* Remove inputs to layers

## Basic Usage

```julia
Pkg.add("https://github.com/DrChainsaw/NaiveNASflux.jl")
```

While NaiveNASflux does not come with any built in search policies, it is still possible to do some cool stuff with it. Below is a very simple example of parameter pruning of a model trained on the `xor` function.

First we need some boilerplate to create the model and do the training:

```julia
using NaiveNASflux, Test
import Flux: train!, mse
import Random
Random.seed!(666)
niters = 100

# First lets define a simple architecture
dense(in, outsize, act) = mutable(Dense(nout(in),outsize, act), in, layerfun=ActivationContribution)

invertex = inputvertex("input", 2, FluxDense())
layer1 = dense(invertex, 32, relu)
layer2 = dense(layer1, 1, sigmoid)
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
nprune = 5

# Prune randomly selected neurons
pruned_random = copy(original)
Δnin(pruned_random.outputs[], rand(1:nout(layer1), nout(layer1) - nprune))
apply_mutation(pruned_random)

# Determine which order to prune neurons according to the metric computed by ActivationContribution.
pruneorder = sortperm(neuron_value(layer1))

# Prune the least valuable neurons according to the metric in ActivationContribution
pruned_least = copy(original)
Δnin(pruned_least.outputs[], pruneorder[nprune:end])
apply_mutation(pruned_least)

# Prune the most valuable neurons according to the metric in ActivationContribution
pruned_most = copy(original)
Δnin(pruned_most.outputs[], pruneorder[1:end-nprune])
apply_mutation(pruned_most)

# Can I have my free lunch now please?!
@test loss(pruned_most)(x, y) > loss(pruned_random)(x, y) > loss(pruned_least)(x, y) > loss(original)(x, y)

# The metric calculated by ActivationContribution is actually quite good (in this case).
@test loss(pruned_least)(x, y) ≈ loss(original)(x, y) rtol = 1e-5
```

Also check out the basic usage of [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl) for more examples.

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
