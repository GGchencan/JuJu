# JuJu
This is the first AI model implemented by Julia.


## Note ---- how to add custom grad

- use marco `@grad` 
  
  1. a short piece of program(to real)

  ```julia
  using Flux: @grad, TrackedReal, track

  data(x::TrackedReal) = x.data
  tracker(x::TrackedReal) = x.tracker

  function f(a) = ...

  @grad f(a::Real) = f(data(a)), Δ -> (Δ * $da)
  f(a::TrackedReal) = track(f, a) 

  ```

  2. to some function like softmax, the example extracted from `Flux/tracker/array.jl`

  ```julia
  softmax(xs::TrackedArray) = track(softmax, xs)

  @grad softmax(xs) = softmax(data(xs)), Δ -> (nobacksies(:softmax, ∇softmax(data(Δ), data(xs))),)
  ```
