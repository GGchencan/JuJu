function PredictLabelOneDim(Data)
    """
    return the index of largest number in data
    """
    Len = size(Data, 1)
    Index = 0
    MaxVal = 0
    for i = 1:Len
        if Data[i] > MaxVal
            Index = i
            MaxVal = Data[i]
        end
    end
    if Index == 10
        Index = 2
    end
    return Index
end

function PredictLabel(data)
    """
    Input: Array(BatchSize, SeqLen, LabelSize), float64
    Output: Array(BatchSize, SeqLen),Int64
    """
    BatchSize = size(data, 1)
    SeqLen = size(data, 2)
    Predict = []
    for i in 1:BatchSize
        Tmp = []
        for j in 1:SeqLen
            Tmp = vcat(Tmp,PredictLabelOneDim(data[i,j,:]))
        end
        if i == 1
            Predict = Tmp
        else
            Predict = hcat(Predict, Tmp)
        end
    end
    return convert(Array{Int8,2}, Predict')
end

function Accuracy(Predict, Truth)
    """
    Predict (BatchSize, SeqLen),Int64
    Truth (BatchSize, SeqLen),Int64
    """
    BatchSize = size(Predict, 1)
    #println(BatchSize)
    SeqLen = size(Predict, 2)
    TrueCount = 0

    for i in 1:BatchSize
        flag = true
        for j in 1:SeqLen
            #print("%%%%%%%%%%j")
            #print(j)
            #println(Predict[i])
            if Predict[i,j] != Truth[i,j]
                flag = false
                break
            end
        end
        TrueCount = TrueCount + flag
    end
    return float(TrueCount)/BatchSize
end
