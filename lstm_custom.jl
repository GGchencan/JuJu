
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# Stateful recurrence

"""
    Recur(cell)

`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:

    h, y = cell(h, x...)

For example, here's a recurrent network that keeps a running total of its inputs.

```julia
accum(h, x) = (h+x, x)
rnn = Flux.Recur(accum, 0)
rnn(2) # 2
rnn(3) # 3
rnn.state # 5
rnn.(1:10) # apply to a sequence
rnn.state # 60
```
"""
#

_truncate(x::AbstractArray) = Tracker.data(x)
_truncate(x::Tuple) = _truncate.(x)

"""
    truncate!(rnn)

Truncates the gradient of the hidden state in recurrent layers. The value of the
state is preserved. See also `reset!`.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to

    rnn.state = Tracker.data(rnn.state)
"""
truncate!(m) = prefor(x -> x isa Recur && (x.state = _truncate(x.state)), m)

"""
    reset!(rnn)

Reset the hidden state of a recurrent layer back to its original value. See also
`truncate!`.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to

    rnn.state = hidden(rnn.cell)
"""




mutable struct LSTMCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
  c::V
end

function LSTMCell(in::Integer, out::Integer;
                  init = Flux.glorot_uniform)
  cell = LSTMCell(param(init(out*4, in)), param(init(out*4, out)), param(zeros(out*4)),
                  param(init(out)), param(init(out)))
  cell.b.data[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell)((h, c), x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return (h′, c), h′
end

hidden(m::LSTMCell) = (m.h, m.c)

Flux.@treelike LSTMCell

Base.show(io::IO, l::LSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")






mutable struct BiLSTMCell{T}
  forward::T
  backward::T
end

function BiLSTMCell(in::Integer,out::Integer;init = Flux.glorot_uniform)
  cell = BiLSTMCell(LSTMCell(in, out), LSTMCell(in, out))
  return cell
end


# Assuming that the h is shaped like (N, 2*h), c is shaped like (N, 2*h), x is shaped like((N, in),(N, in))
function (m::BiLSTMCell)((h,c),x)
  out_dim = size(h, 2) / 2
  out_dim = Int(out_dim)
  in_dim = size(x[1],2)
  forward_h = h[:, gate(out_dim, 1)]
  forward_h = permutedims(forward_h, [2, 1])
  backward_h = h[:, gate(out_dim, 2)]
  backward_h = permutedims(backward_h, [2, 1])
  forward_c = c[:, gate(out_dim, 1)]
  forward_c = permutedims(forward_c, [2, 1])
  backward_c = c[:, gate(out_dim, 2)]
  backward_c = permutedims(backward_c, [2, 1])
  x_forward = permutedims(x[1], [2, 1])
  x_backward = permutedims(x[2], [2, 1])
  (forward_h_new, forward_c_new), _ = m.forward((forward_h, forward_c), x_forward)
  (backward_h_new, backward_c_new), _ = m.backward((backward_h, backward_c), x_backward)
  forward_h_new = permutedims(forward_h_new, [2,1])
  forward_c_new = permutedims(forward_c_new, [2,1])
  backward_h_new = permutedims(backward_h_new, [2,1])
  backward_c_new = permutedims(backward_c_new, [2,1])
  h_new = hcat(forward_h_new, backward_h_new)
  c_new = hcat(forward_c_new, backward_c_new)
  return (h_new, c_new), h_new
end
hidden(m::BiLSTMCell) = (vcat(m.forward.h, m.backward.h), vcat(m.forward.c, m.backward.c))
Flux.@treelike BiLSTMCell


Base.show(io::IO, l::BiLSTMCell) =
  print(io, "BiLSTMCell(", size(l.forward.Wi, 2), ", ", size(l.forward.Wi, 1)÷4, ")")



mutable struct MyRecur{T}
  cell::T
  init
  state
end

MyRecur(m, h = hidden(m)) = MyRecur(m, h, h)

# ... combines many arguments into one argument in function definitions as a tuple
# ... splits one argument into many different arguments in function calls
function (m::MyRecur)(xs)
  if length(size(m.state[1]))==1
    newstate1 = add_dim(m.state[1])
    newstate2 = add_dim(m.state[2])
    m.state = (newstate1, newstate2)
  end
  h, y = m.cell(m.state,xs )
  m.state = h
  return y
end

Flux.@treelike MyRecur cell, init

Base.show(io::IO, m::MyRecur) = print(io, "MyRecur(", m.cell, ")")

RawBiLSTM(a...; ka...) = MyRecur(BiLSTMCell(a...; ka...))

mutable struct MyBiLSTM
  cell
end

MyBiLSTM(in::Integer,out::Integer)= MyBiLSTM( RawBiLSTM(in,out))

add_dim(x) = reshape(x, (1, size(x)...))
function (m::MyBiLSTM)(data; batch_first::Bool=true)
  # Assuming that data is shaped like (batch, seq_len, dim) if batch_first is True or the shape is like (seq_len, batch, dim)
  if batch_first
    data = permutedims(data,[2,1,3])
  end
  m.cell.state = m.cell.init
  seq_len = size(data, 1)
  data_forward = [data[i,:,:] for i in 1:1:seq_len]
  data_backward = [data_forward[i] for i in seq_len:-1:1]
  dim = size(m.cell.cell.forward.b.data,1) / 4
  dim = convert(Int, dim)
  forward = []
  backward = []
  for data in zip(data_forward, data_backward)
    output =  m.cell(data)
    output_forward = add_dim(output[:, 1:1:dim])

    output_backward = add_dim(output[:,dim+1:1:2*dim])
    push!(forward, output_forward)
    push!(backward, output_backward)
  end
  backward_inverse = []
  len = length(backward)
  for _ in range(1,stop = len)
    push!(backward_inverse, pop!(backward))
  end
  outputs = []
  for data in zip(forward, backward_inverse)
      push!(outputs, cat(data...; dims=3))
  end
  y =  cat(outputs..., dims=1)
  if batch_first
    y = permutedims(y, [2,1,3])
  end
  return y
end

Flux.@treelike MyBiLSTM
