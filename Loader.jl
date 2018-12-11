using Random
using DelimitedFiles
using ResumableFunctions


"""
n*m -> n*m*class_num
"""

function onehot(Array, ClassNum, Idx)
    N = size(Array)[1]
    M = size(Array)[2]
    Result = zeros(Float32, (N, Idx, ClassNum))
    for i in 1:N
        for j in 1:Idx
            Result[i, j, Array[i, j]] = 1
        end
    end
    return Result
end


"""
this generator returns one epoch of Minibatches
Steps = Int(NumExamples/BatchSize)
"""

@resumable function one_epoch(DataDict, BatchSize, NumWords, NumLabels) :: Tuple
    NumExamples = size(DataDict['x'])[1]
    Steps = convert(Int, floor(NumExamples / BatchSize))
    RandomIdx = Random.randperm(NumExamples)

    for i in 1:Steps
        MiniIdx = RandomIdx[(i - 1) * BatchSize + 1:i * BatchSize]
        MiniX = DataDict['x'][MiniIdx,:]
        RowSum = sum(MiniX, dims = 1)
        N1 = size(RowSum)[1]
        M1 = size(RowSum)[2]
        Idx = M1
        Null = 2 * BatchSize
        for i in 0 : M1 - 1
            if RowSum[N1, Idx] != Null
                break
            end
            Idx -= 1
        end
        @yield (onehot(DataDict['x'][MiniIdx,:], NumWords, Idx), onehot(DataDict['y'][MiniIdx,:], NumLabels, Idx))
    end
end


"""
NumWords means how many words in our dict, NumLabels means how many labels
"""

@resumable function minibatches(DataDict, BatchSize, NumWords, NumLabels, Steps) :: Tuple
    NumExamples = size(DataDict['x'])[1]
    for i in 1:Steps
        RandomIdx = Random.randperm(NumExamples)[1:BatchSize]
        MiniX = DataDict['x'][RandomIdx,:]
        RowSum = sum(MiniX, dims = 1)
        N1 = size(RowSum)[1]
        M1 = size(RowSum)[2]
        Idx = M1
        Null = 2 * BatchSize
        for i in 0 : M1 - 1
            if RowSum[N1, Idx] != Null
                break
            end
            Idx -= 1
        end
        @yield (onehot(DataDict['x'][RandomIdx,:], NumWords, Idx), onehot(DataDict['y'][RandomIdx,:], NumLabels, Idx))
    end
end


function read_file()

    """
    special symbols
    EOF:1
    pad:2
    less than min frequence:3

    containg two Arrays : 'x' contains featuers, 'y' contains labels
    please note the index in julia strating at 1, the same as matlab.
    a=[1, 2]. a[1]=1 and a[2]=2
    """

    TrainingDict = Dict()
    TestingDict = Dict()
    TrainingDict['x'] = DelimitedFiles.readdlm("trn_x.txt", ' ', Int)
    TrainingDict['y'] = DelimitedFiles.readdlm("trn_y.txt", ' ', Int)
    TestingDict['x'] = DelimitedFiles.readdlm("test_x.txt", ' ', Int)
    TestingDict['y'] = DelimitedFiles.readdlm("test_y.txt", ' ', Int)

    WordDict = Dict()
    LabelDict = Dict()
    open("word_dict.txt") do f
        for l in eachline(f)
            Arr = split(l)
            WordDict[Arr[1]] = parse(Int, Arr[2])
        end
    end
    open("label_dict.txt") do f
        for l in eachline(f)
            Arr = split(l)
            LabelDict[Arr[1]] = parse(Int, Arr[2])
        end
    end
    return TrainingDict, TestingDict, WordDict, LabelDict
end

"""
 BatchSize = 100

 TrainingDict, TestingDict, WordDict, LabelDict = read_file()
 generator = one_epoch(TrainingDict, BatchSize, length(WordDict), length(LabelDict))
 for Arr in generator
     mini_trn_x = Arr[1]
     mini_trn_y = Arr[2]

     print("size of input x matrix: ", size(mini_trn_x), " it has type", typeof(mini_trn_x), '\n', "size of input y matrix: ", size(mini_trn_y), " it has type", typeof(mini_trn_y), '\n')

 end
"""
