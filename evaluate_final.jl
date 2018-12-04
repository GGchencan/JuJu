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
    #tagString = ["O", "B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG","B-MISC","I-MISC"]
    tagString = ["B-ORG", "O", "B-MISC", "B-PER","I-PER","B-LOC","I-ORG","I-MISC","I-LOC"]
    evaluate = zeros(3) #correct trueChunks predChunks
    #println(evaluate)
    correctChunk = "None"
    prevPredPrefix = "O"
    prevTruePrefix = "O"
    prevTrueTag =  "O"
    prevPredTag  =  "O"
    for i in 1:length(predictSeqs)
        #println(length(predictSeqs))
        trueTag = tagString[trueSeqs[i]]
        predTag = tagString[predictSeqs[i]]
        #println(    trueTag)
        truePrefix , trueType = splitTag(trueTag)
        PredPrefix , predType = splitTag(predTag)

        #println(trueType)
        #println("**************")
        if correctChunk != "None"
            trueEnd = endOfChunk(prevTruePrefix, truePrefix)
            predEnd = endOfChunk(prevPredPrefix, PredPrefix)
            #println(predEnd)
            #println(trueEnd)
            #println("**************")
            #println(correctChunk)



            if predEnd & trueEnd & (prevTrueTag== prevPredTag) & (trueTag  == predTag)
            #if  trueEnd
            """
            println("**************")
            print("i=  ")
            println(i-1)
            println(trueSeqs[i-1])
            println(predictSeqs[i-1])
            println(prevTrueTag)
            #println("**************")
            println( prevPredTag)
            print("i=  ")
            println(i)
            println(trueSeqs[i])
            println(predictSeqs[i])
            #println("**************")
            println(trueTag)
            println(predTag)
            println("**************")
            """
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
        prevTrueTag = trueTag
        prevPredTag  = predTag
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
    return (P, R, F_1,evaluate[1],evaluate[2],evaluate[3])
end



#tagString = ["B-ORG", "O", "B-MISC", "B-PER","I-PER","B-LOC","I-ORG","I-MISC","I-LOC"]
trueSeqs =    [1 7 2 3 8;2 6 9 2 3]'
predictSeqs = [1 1 2 3 3;2 6 6 2 3]'
f1 =  countChunks(trueSeqs,predictSeqs)
println(f1)
