# NaiveNASflux

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://DrChainsaw.github.io/NaiveNASflux.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://DrChainsaw.github.io/NaiveNASflux.jl/dev)
[![Build status](https://github.com/DrChainsaw/NaiveNASflux.jl/workflows/CI/badge.svg?branch=master)](https://github.com/DrChainsaw/NaiveNASflux.jl/actions)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASflux.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASflux-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASflux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASflux.jl)

NaiveNASflux uses [NaiveNASlib](https://github.com/DrChainsaw/NaiveNASlib.jl) to enable mutation operations of arbitrary [Flux](https://github.com/FluxML/Flux.jl) computation graphs. It is designed with Neural Architecture Search (NAS) in mind, but can be used for any purpose where doing changes to a model is desired.

Note that NaiveNASflux does not have any functionality to search for a model architecture. Check out [NaiveGAflux](https://github.com/DrChainsaw/NaiveGAflux.jl) for a simple proof of concept.  

## Basic Usage

```julia
]add NaiveNASflux
```

See [documentation](https://DrChainsaw.github.io/NaiveNASflux.jl/stable) for usage instructions.

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
