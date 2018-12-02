using Flux
using Flux: onehot, onehotbatch, crossentropy, reset!, throttle, @epochs, @show
using Flux.Optimise: SGD
using BSON: @save, @load
using Plots

include("Loader.jl")
include("lstm_custom.jl")
include("PredictLabel.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
epoch_size = 1
batch_size = 100
ebemd_size = 100
hidden_size = 100
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


#data = Minibatches(traindata, batch_size, dic_size, class_num, 1000)
data = One_Epoch(traindata, batch_size, dic_size, class_num)
#LossHistory = []
for d in data
    """
    add test while training
    """
    x = d[1]
    output = UpperDim(class_num, batch_size)(model(x))
    predict = PredictLabel(output.data)
    truth = PredictLabel(d[2])
    print("accuray is \n")
    print(countChunks(predict, truth))

    Flux.train!(loss, [d], opt)
    #LossHistory = vcat(LossHistory, loss(d[1],d[2]).data)
end


"""

plotly() # Choose the Plotly.jl backend for web interactivity
plot(LossHistory,linewidth=2,title="Train Loss")
"""

"""
img saved in:
C:\Users\v-checan\AppData\Local\Temp
"""

#test = Minibatches(testdata, batch_size, dic_size, class_num, 1000)
test = One_Epoch(testdata, batch_size, dic_size, class_num)
for d in test
    x = d[1]
    """
    print("here is model info")
    print(size(model(x)))
    print(typeof(model(x)))

    print("after tranforming")
    print(size(UpperDim(class_num, batch_size)(model(x))))
    print(typeof(UpperDim(class_num, batch_size)(model(x))))


    print("label info")
    print(size(d[2]))
    print(typeof(d[2]))
    break
    """

    output = UpperDim(class_num, batch_size)(model(x))
    predict = PredictLabel(output)
    print(Accuracy(predict, d[2]))
    print("\n")
    print("*****************************************************")
end

