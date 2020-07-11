# Stuff that runs once to make sure mutable layers work with Flux.functor

# Modified version of Flux.functor for mutable structs which are mutable mainly because they are intended to be wrapped in MutationVertices which in turn are not easy to create in the manner which Flux.functor is designed.
function mutable_makefunctor(m::Module, T, fs = functor_fields(T))
  @eval m begin
    Flux.functor(::Type{<:$T}, x) = ($([:($f=x.$f) for f in fs]...),),
    # Instead of creating a new T, we set all fields in fs of x to y (since y is the fields returned in line above)
    function(y)
      $([:(x.$f = y[$i]) for (i, f) in enumerate(fs)]...)
      return x
    end
  end
end

function mutable_functorm(T, fs = nothing)
  fs == nothing || isexpr(fs, :tuple) || error("@functor T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(mutable_makefunctor(@__MODULE__, $(esc(T)), $(fs...)))
end

macro mutable_functor(args...)
  mutable_functorm(args...)
end

functor_fields(T) = fieldnames(T)

for subtype in filter(st -> st !== LazyMutable, subtypes(AbstractMutableComp))
  @mutable_functor subtype
end
