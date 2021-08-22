# Layer Wrappers

NaiveNASflux wraps Flux layers in mutable wrapper types by default so that the vertex operations can be mutated without having to recreate the whole model. Additional wrappers which might be useful are described here.

```@docs
ActivationContribution
LazyMutable
```
