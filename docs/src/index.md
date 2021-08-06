# Introduction

NaiveNASflux is an extension of [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl) which adds primitives for [Flux](https://github.com/FluxML/Flux.jl) layers so that they can be used in a computation graph which NaiveNASlib can modify. Apart from this, it adds very little new functionality.

## Reading Guideline

Due to how NaiveNASflux just glues Flux and NaiveNASlib, most of the things one can use NaiveNASflux for is described in the [documentation for NaiveNASlib](https://DrChainsaw.github.io/NaiveNASlib.jl/stable).

The [Quick Tutorial](@ref) gives a quick overview of what using NaiveNASflux might look like while the [Model Pruning Example](@ref) show lightweight usage of NaiveNASflux without bringing in full fledged neural architecture search.

The API reference is split up into categories in an attempt to make it easy to answer "how do I achieve X?"-type questions.