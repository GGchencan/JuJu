"""
Usage:
example0:
trueSeqs =    [1 7 2 3 8 10 10 10;2 6 9 2 3 10 10 10]
predictSeqs = [1 1 2 3 3 3 10 10 ;2 6 6 2 3 4 10 10]
f1 = countChunks(trueSeqs,predictSeqs)
output
processed some tokens with 4.0 phrases; found: 7.0 phrases; correct: 1.0
precision: 14.29%;  recall:  25.00%;  FB1:  0.18


example1:
input:
#tagString = ["2 B-ORG", "2 O", "3 B-MISC", "4 B-PER","5 I-PER","6 B-LOC","7 I-ORG","8 I-MISC","9 I-LOC"]
trueSeqs =    [1 7 2  3 8 10 10 10 10 10;2 6 9 2 3 1  7  10 10 10]
predictSeqs = [1 1 10 3 3 1  10 3   3 2 ;2 6 6 2 3 10 10 10 10 10]
f1 = countChunks(trueSeqs,predictSeqs)

ouput:
processed some tokens with 5.0 phrases; found: 7.0 phrases; correct: 1.0
precision: 14.29%;  recall:  20.00%;  FB1:  0.17

example2:
input:
trueSeqs =    [10 10 10 10 10;2 10 10 10 10]
predictSeqs = [1 1 10 3 3 1  ;2 6 6 2 3 10]
f1 = countChunks(trueSeqs,predictSeqs)

output:
precision: 100.00%;  recall:  60.00%;  FB1:  0.75
processed some tokens with 0.0 phrases; found: 0.0 phrases; correct: 0

example3:
input:
trueSeqs =    [1 1 1 1 1 10 10 10 ;2 2 2 2 2 10 10 10]
predictSeqs = [1 2 1 2 1 10 10 10 ;2 2 2 2 2 10 10 10 ]
f1 = countChunks(trueSeqs,predictSeqs)
output:
processed some tokens with 5.0 phrases; found: 3.0 phrases; correct: 3.0
precision: 100.00%;  recall:  60.00%;  FB1:  0.75

example4:
input:
trueSeqs =    [1 1 1 1 1 ;2 2 2 2 2;1 7 2 3 8;2 6 9 2 3;1 7 2 6 2;2 2 2 2 3]'
predictSeqs = [1 2 1 2 1 ;2 2 2 2 2 ;1 1 2 3 3;2 6 6 2 3;1 2 3 4 4;5 7 8 9 3]'

output:
processed some tokens with 12.0 phrases; found: 19.0 phrases; correct: 5.0
precision: 26.32%;  recall:  41.67%;  FB1:  0.32

example5：
input
trueSeqs =    [1 1 10 10 10 ;2 10 10 10 10]
predictSeqs = [1 1 10 10 10 ;2 10 10 10 10 ]
output

(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)

"""

function splitTag(chunkTag)
    if chunkTag == "O"
        return ("O","None")
    end
    return split(chunkTag,'-')
end

function startOfChunk(prevTag,prevType,tag,type )
    chunkStart = false
    if(tag == "B")   chunkStart = true  end
    if((prevTag == "O") & (tag == "I")) chunkStart = true  end
    if((tag != "O") & (prevType != type))  chunkStart = true  end
    return chunkStart
end

function endOfChunk(prevTag, prevType,tag,type )
    chunkEnd = false
    chunkEnd = (((prevTag == "B") & (tag == "B"))
                |((prevTag == "I") & (tag == "B"))
                | ((prevTag == "B") & (tag == "O"))
                |((prevTag == "I") & (tag == "O"))
                | ((tag == "O") & (prevType != type)))
  return chunkEnd
end

function countChunks(trueSeqs, predictSeqs)
    tagString = ["B-ORG", "O", "B-MISC", "B-PER","I-PER","B-LOC","I-ORG","I-MISC","I-LOC","O"]
    evaluate = zeros(3)

    startFlage= false
    prevPredPrefix = "O"
    prevTruePrefix = "O"
    prevPredType = "O"
    prevTrueType = "O"
    prevTrueTag =  "O"
    prevPredTag  =  "O"

    N = size(trueSeqs,1)
    Col = size(trueSeqs,2)
    for i in 1:N
        if findmax(trueSeqs[1,:])[1] < 10
            M = Col+1
        else
            M = findmax(trueSeqs[i,:])[2]
        end
        startFlage= false
        prevPredPrefix = "O"
        prevTruePrefix = "O"
        prevPredType = "O"
        prevTrueType = "O"
        prevTrueTag =  "O"
        prevPredTag  =  "O"
        for j in 1:M-1
            trueTag = tagString[trueSeqs[i,j]]
            predTag = tagString[predictSeqs[i,j]]
            truePrefix , trueType = splitTag(trueTag)
            PredPrefix , predType = splitTag(predTag)

            if startFlage ==  true
                trueEnd = endOfChunk(prevTruePrefix,prevTrueType ,truePrefix,trueType)
                predEnd = endOfChunk( prevPredPrefix,prevPredType,PredPrefix, predType)
                if predEnd & trueEnd & (prevTrueTag== prevPredTag)
                    evaluate[1] += 1
                    startFlage= false
                elseif (predEnd != trueEnd) | (trueType != predType)
                    startFlage= false
                end
            end
            trueStart = startOfChunk(prevTruePrefix,prevTrueType ,truePrefix,trueType)
            predStart = startOfChunk( prevPredPrefix,prevPredType,PredPrefix, predType)

            if trueStart & predStart & (trueType == predType)
                startFlage = true
            end
            if trueStart
                evaluate[2] += 1
            end
            if predStart
                evaluate[3] += 1
            end

            prevTruePrefix  = truePrefix
            prevPredPrefix  = PredPrefix
            prevTrueType = trueType
            prevPredType = predType
            prevTrueTag = trueTag
            prevPredTag  = predTag
        end
        if startFlage == true
            evaluate[1] += 1
        end
    end

    if (evaluate[1] == 0)
        println("processed some tokens with $(evaluate[2]) phrases; found: $(evaluate[3]) phrases; correct: 0")
        return(0,0,0,0,0,0)
    end

    P = evaluate[1]/evaluate[3]
    R = evaluate[1]/evaluate[2]
    F_1 = 2*P*R / (P+R)
    println("**********************")
    println("processed some tokens with $(evaluate[2]) phrases; found: $(evaluate[3]) phrases; correct: $(evaluate[1])")
    println( "precision: $(@sprintf("%.2f",P*100))%;  recall:  $(@sprintf("%.2f", R*100))%;  FB1:  $(@sprintf("%.2f",F_1))")
    return (P, R, F_1,evaluate[1],evaluate[2],evaluate[3])

end

trueSeqs =    [1 7 2 3 8 10 10 10;2 6 9 2 3 10 10 10]
predictSeqs = [1 1 2 3 3 3 10 10 ;2 6 6 2 3 4 10 10]
f1 = countChunks(trueSeqs,predictSeqs)
