using Flux
using Flux: onehot, onehotbatch, crossentropy, reset!, throttle, @epochs, @show
using Flux.Optimise: SGD
using BSON: @save, @load
include("Loader.jl")
include("BiLSTM.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
epoch_size = 1
batch_size = 20
ebemd_size = 10
hidden_size = 10
input_fn = "a"
min_freq = 0
model_fn = "b"

traindata, testdata, dic, label_dic = Readfile(input_fn, min_freq)
dic_size = length(dic)
class_num = length(label_dic)

# println(dic_size)
# println(class_num)
"""
Get train/test batch data
batch_size * seq_len * dic_size
"""

trainX, trainY = Minibatch(traindata, batch_size, dic_size, class_num)
testX, testY = Minibatch(testdata, batch_size, dic_size, class_num)

function LowerDim(dim)
    x -> reshape(x, (:, dim))'
end

function UpperDim(ebemd_dim, batch_size)
    x -> reshape(x', (batch_size, :, ebemd_dim))
end

function ChangeDim(dim)
    x -> permutedims(x, dim)
end

function BILSTM(EmbeddingSize, HiddenSize)
    x -> BiLSTM(x, EmbeddingSize, HiddenSize)
end

model = Chain(
    LowerDim(dic_size),
    Dense(dic_size, ebemd_size),
    UpperDim(ebemd_size, batch_size),
    ChangeDim([2, 3, 1]),
    BILSTM(ebemd_size, hidden_size),
    ChangeDim([3, 1, 2]),
    LowerDim(hidden_size),
    Dense(hidden_size, class_num),
    softmax)


function loss(x, y)
    l = crossentropy(model(x), LowerDim(class_num)(y))
    # print('loss ', l)
    Flux.truncate!(model)
    # @show l
    return l
end


function save_model(model_fn)
    if model_fn == false
        @save "model-$(now()).bson" model
    else
        @save model_fn model
    end
    println("saving model complete.")
end


function evalcb_batch()
    save_model(false)
end


function load_model(checkpoint_fn)
    @load checkpoint_fn model
end

# function init_model()
#
# end


lr = 0.1
opt = SGD(params(model), lr)

for i = 1 : epoch_size
    Flux.train!(loss, zip(allX, allY), opt, cb = throttle(evalcb_batch, 60))
    # println(i, "epoch loss is  ", sum(loss.(testX, testY)))
end

# @epochs epoch_size Flux.train!(loss, zip(trainX, trainY), opt, cb = throttle(evalcb_batch, 60))

save_model(model_fn)
