using Flux
using Flux: onehot, onehotbatch, crossentropy, reset!, throttle, @epochs, @show
using Flux.Optimise: SGD
# using BSON: @save, @load
# using Plots

include("Loader.jl")
include("lstm_custom.jl")
include("PredictLabel.jl")
include("evaluate_final.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
epoch_size = 10
batch_size = 100
ebemd_size = 50
hidden_size = 300
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
    MyBiLSTM(ebemd_size, hidden_size),
    LowerDim(hidden_size * 2),
    Dense(hidden_size * 2, class_num),
    softmax
    )


function loss_with_mask(x, y)
    lower_y = LowerDim(class_num)(y)
    dimr = size(lower_y)[1]
    dimc = size(lower_y)[2]
    w = ones(dimr, dimc)
    w[dimr,:] = zeros(dimc)
    l = crossentropy(model(x), lower_y; weight = w)
    # print('loss ', l)
    Flux.truncate!(model)
    @show(l)
    return l
end

function loss(x, y)
    lower_y = LowerDim(class_num)(y)
    l = crossentropy(model(x), lower_y)
    # print('loss ', l)
    Flux.truncate!(model)
    @show(l)
    return l
end

function load_model(checkpoint_fn)
    load_cpu(model, checkpoint_fn)
end


lr = 0.005
opt = SGD(params(model), lr)

test = One_Epoch(testdata, batch_size, dic_size, class_num)
testd = test(1)

for i = 1 : epoch_size
    println("epoch ", i)
    data = One_Epoch(traindata, batch_size, dic_size, class_num)
    for d in data
        Flux.train!(loss, [d], opt)
        x = testd[1]
        output = UpperDim(class_num, batch_size)(model(x))
        predict = PredictLabel(output.data)
        truth = PredictLabel(testd[2])
        print("accuray is \n")
        print(countChunks(truth',predict'))
    end
    save_cpu(model, "model")
end
