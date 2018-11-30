"""
Usage:
include("Evaluate.jl")
True = ["B-Pre I-Pre O O","O O  B-Pre B-Loc"]
Predict = ["B-Pre I-Pre O O","O O  B-Pre I-Pre"]
f1 = main(True,Predict)
println(f1)

"""



"""
split chunk tag into IOBES prefix and chunk_type
e.g.
B-PER -> (B, PER)
O -> (O, None)
"""

function splitTag(chunkTag)
    if chunkTag == "O"
        return ("O","None")
    end
    return split(chunkTag,'-')
end
#True = [["B-Pre" ,"I-Pre", "O", "O"],["O", "O" , "B-Pre" ,"B-Loc"]]
Predict = [["B-Pre" "i-Pre" "O" "O"],["O  O  B-Pre I-Pre"]]

"""
checks if a chunk started between the previous and current word;
"""
function startOfChunk(prevTag, tag)
  chunkStart = (((prevTag == "B") & (tag == "B"))
    |((prevTag == "I") & (tag == "B"))
    | ((prevTag == "O") & (tag == "B")) )
return chunkStart
end
"""
checks if a chunk ended between the previous and current word;
"""
function endOfChunk(prevTag, tag)

  chunkEnd = (((prevTag == "B") & (tag == "B")) |((prevTag == "I") & (tag == "B"))
              | ((prevTag == "O") & (tag == "B")) | ((prevTag == "B") & (tag == "O"))
               |((prevTag == "I") & (tag == "O")) | ((prevTag == "O") & (tag == "O")))
  return chunkEnd

end

#function  countChunks(trueSeqs,predictSeqs)
function countChunks(trueSeqs, predictSeqs)
    evaluate = zeros(3) #correct trueChunks predChunks
    correctChunk = "None"
    prevPredPrefix = "O"
    prevTruePrefix = "O"
    for i in 1:length(predictSeqs)
        trueTag = trueSeqs[i]
        predTag = predictSeqs[i]
        truePrefix , trueType = splitTag(trueTag)
        PredPrefix , predType = splitTag(predTag)
        #println(trueType)
        #println("**************")
        if correctChunk != "None"
            trueEnd = endOfChunk(prevTruePrefix, truePrefix)
            predEnd = endOfChunk(prevPredPrefix, PredPrefix)
            #println(trueEnd)
            #println("**************")
            if predEnd & trueEnd
                evaluate[1] += 1
                correctChunk = "None"
            elseif (predEnd != trueEnd) | (trueType != predType)
                correctChunk = "None"
            end
        end
        trueStart = startOfChunk(prevTruePrefix, truePrefix)
        predStart = startOfChunk(prevPredPrefix, PredPrefix)
        #println(trueStart)
        #println("**************")
        if trueStart & predStart & (trueType == predType)
            correctChunk = trueType
        end
        if trueStart
            evaluate[2] += 1
        end
        if predStart
            evaluate[3] += 1
        end
        prevTruePrefix  = truePrefix
        prevPredPrefix  = PredPrefix
    end
    if correctChunk != "None"
        evaluate[1] += 1
    end
    #println(evaluate)
    P = evaluate[1]/evaluate[3]

    #println("**************")
    R = evaluate[1]/evaluate[2]
    #println("**************")
    F_1 = 2*P*R / (P+R)
    #println(F_1)
    return F_1
end

"""
Split string
"""
function splitString(iter)
    seqs = [];
    for line in iter
        seq = split(line)
        #println(length(seq))
        for i in 1:length(seq)
            push!(seqs,seq[i])
        end
    end
#    println(seqs)
    return seqs
end

"""
checks if a chunk started between the previous and current word;
"""
function startOfChunk(prevTag, tag)
  chunkStart = (((prevTag == "B") & (tag == "B"))
    |((prevTag == "I") & (tag == "B"))
    | ((prevTag == "O") & (tag == "B")) )
return chunkStart

"""
checks if a chunk ended between the previous and current word;
"""
function endOfChunk(prevTag, tag)
  chunkEnd = (((prevTag == "B") & (tag == "B")) |((prevTag == "I") & (tag == "B"))
              | ((prevTag == "O") & (tag == "B")) | ((prevTag == "B") & (tag == "O"))
               |((prevTag == "I") & (tag == "O")) | ((prevTag == "O") & (tag == "O")))
  return chunkEnd
end
end

function main(True,Predict)
    trueSeqs = splitString(True)
    predictSeqs = splitString(Predict)
    #println(trueSeqs)
    #println("**************")
    #println(predictSeqs)
    #println("**************")

    return countChunks(trueSeqs,predictSeqs )
    #println(f1)
end
