"""
Usage:
trueSeqs =    [1 7 2 3 8;2 6 9 2 3]'
predictSeqs = [1 1 2 3 3;2 6 6 2 3]'
f1 = countChunks(trueSeqs,predictSeqs)
println(f1)



example1:
input:
#tagString = ["2 B-ORG", "2 O", "3 B-MISC", "4 B-PER","5 I-PER","6 B-LOC","7 I-ORG","8 I-MISC","9 I-LOC"]
trueSeqs =    [1 7 2 3 8;2 6 9 2 3]'
predictSeqs = [1 1 2 3 3;2 6 6 2 3]'

ouput:
(0.14285714285714285, 0.25, 0.18181818181818182, 1.0, 4.0, 7.0)

example2:
input:
trueSeqs =    [1 7 2 6 2;2 2 2 2 3]'
predictSeqs = [1 2 3 4 4;5 7 8 9 3]'

output:
(0.2, 0.3333333333333333, 0.25, 1.0, 3.0, 5.0)

example3:
input:
trueSeqs =    [1 1 1 1 1 ;2 2 2 2 2]'
predictSeqs = [1 2 1 2 1 ;2 2 2 2 2 ]'
output:
(0.3333333333333333, 0.2, 0.25, 3.0, 5.0, 3.0)

input:
trueSeqs =    [1 1 1 1 1 ;2 2 2 2 2;1 7 2 3 8;2 6 9 2 3;1 7 2 6 2;2 2 2 2 3]'
predictSeqs = [1 2 1 2 1 ;2 2 2 2 2 ;1 1 2 3 3;2 6 6 2 3;1 2 3 4 4;5 7 8 9 3]'

output:
(0.2, 0.25, 0.22222222222222224, 3.0, 12.0, 15.0)

"""

"""
split chunk tag into IOBES prefix and chunk_type
e.g.
B-PER -> (B, PER)
O -> (O, None)
"""
function splitTag(ChunkTag)
    if ChunkTag == "O"
        return ("O","None")
    end
    return split(ChunkTag,'-')
end


"""
Check if it is the beginning of a Chunk
"""
function startOfChunk(Tag)
 ChunkStart = (Tag == "B")
return ChunkStart
end


"""
checks if a chunk ended between the previous and current word;
"""
function endOfChunk(PrevTag, Tag)
  ChunkEnd = (((PrevTag == "B") & (Tag == "B")) |((PrevTag == "I") & (Tag == "B"))
              | ((PrevTag == "B") & (Tag == "O")) |((PrevTag == "I") & (Tag == "O")))
  return ChunkEnd
end


function countChunks(TrueSeqs, PredictSeqs)
    TagString = ["B-ORG", "O", "B-MISC", "B-PER","I-PER","B-LOC","I-ORG","I-MISC","I-LOC"]
    Evaluate = zeros(3)
    StartFlage= false
    PrevPredPrefix = "O"
    PrevTruePrefix = "O"
    PrevTrueTag =  "O"
    PrevPredTag  =  "O"

    for i in 1:length(PredictSeqs)
        TrueTag = TagString[TrueSeqs[i]]
        PredTag = TagString[PredictSeqs[i]]
        TruePrefix , TrueType = splitTag(TrueTag)
        PredPrefix , PredType = splitTag(PredTag)

        if StartFlage ==  true
            TrueEnd = endOfChunk(PrevTruePrefix, TruePrefix)
            PredEnd = endOfChunk(PrevPredPrefix, PredPrefix)
            if PredEnd & TrueEnd & (PrevTrueTag== PrevPredTag)
                Evaluate[1] += 1
                StartFlage= false
            elseif (PredEnd != TrueEnd) | (TrueType != PredType)
                StartFlage= false
            end
        end

        TrueStart = startOfChunk(TruePrefix)
        PredStart = startOfChunk( PredPrefix)

        if TrueStart & PredStart & (TrueType == PredType)
            StartFlage = true
        end
        if TrueStart
            Evaluate[2] += 1
        end
        if PredStart
            Evaluate[3] += 1
        end

        PrevTruePrefix  = TruePrefix
        PrevPredPrefix  = PredPrefix
        PrevTrueTag = TrueTag
        PrevPredTag  = PredTag
    end

    if StartFlage == true
        Evaluate[1] += 1
    end

    P = Evaluate[1]/Evaluate[3]
    R = Evaluate[1]/Evaluate[2]
    F_1 = 2*P*R / (P+R)

    return (P, R, F_1,Evaluate[1],Evaluate[2],Evaluate[3])
end


TrueSeqs =    [1 1 1 1 1 ;2 2 2 2 2]'
PredictSeqs = [1 2 1 2 1 ;2 2 2 2 2 ]'

F1 = countChunks(TrueSeqs,PredictSeqs)
println(F1)
