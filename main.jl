using Flux
using Flux: onehot, onehotbatch, crossentropy, reset!, throttle, @epochs, @show
using Flux.Optimise: SGD
using BSON: @save, @load
include("Loader.jl")
include("lstm_custom.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
epoch_size = 1
batch_size = 10
ebemd_size = 256
hidden_size = 128
# min_freq = 0
model_fn = "final_model.bson"

traindata, testdata, dic, label_dic = Readfile()
dic_size = length(dic)
class_num = length(label_dic)

println(dic_size)
println(class_num)
"""
Get train/test batch data
batch_size * seq_len * dic_size
"""

# trainX, trainY = Minibatch(traindata, batch_size, dic_size, class_num)
# testX, testY = Minibatch(testdata, batch_size, dic_size, class_num)


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
    # MyBiLSTM(ebemd_size, hidden_size),
    LowerDim(hidden_size * 2),
    Dense(hidden_size * 2, class_num),
    softmax
    )


function loss(x, y)
    l = crossentropy(model(x), LowerDim(class_num)(y))
    # print('loss ', l)
    Flux.truncate!(model)
    @show(l)
    return l
end


function save_model(model_fn)
    if model_fn == false
        @save "model.bson" model
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


lr = 0.1
opt = SGD(params(model), lr)


data = Minibatches(traindata, batch_size, dic_size, class_num, 3)
# for d in data
#     print(typeof(d[1]))
#     print(size(d[2]))
#     print(size(model(d[1])))
#     print(size(LowerDim(class_num)(d[2])))
#     break
# end

# for i = 1 : epoch_size
#     Flux.train!(loss, data, opt)
# end
for d in data
    Flux.train!(loss, [d], opt)
end
@save "mymodel.bson" model
