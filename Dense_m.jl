import Flux
import Flux: @treelike, param
import Flux.Tracker: TrackedArray

struct Dense_m{F,A}
    σ::F
    weight::A
end

function Dense_m(weight::Array)
    return Dense_m(identity, weight')
end

@treelike Dense_m

function (d::Dense_m)(x)
    x = permutedims(x, [2, 3, 1])
    σ, w = d.σ, d.weight
    y = []
    for index in range(1;stop=size(x, 3))
        push!(y, σ.(x[:, :, index] * w))
    end
    return permutedims(cat(y...; dims=3), (3, 1, 2))
end
