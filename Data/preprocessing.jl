using DataStructures

"""
preprocessing.jl
    this file is used for preprocessing the raw data(conll2003)
    the original file is wirtten in python
    this is a simply transfer version (From python to Julia)
"""

function padding(
                arr,
                arr_y,
                label_dict;
                max_length = 52,
                padwidth = 1
                )

        for i in range(1; stop=length(arr))
                diff = max_length - length(arr[i])
                if diff > 0
                        arr[i] = vcat(arr[i]..., [2 for i in range(1; stop=diff)]...)
                        arr_y[i] = vcat(arr_y[i]..., [label_dict["null"] for i in range(1; stop=diff)]...)
                end
        end
        return arr, arr_y
end


word_dict = OrderedDict()
min_word_cnt = 5
word_cnt_dict = OrderedDict()
label_dict = OrderedDict()
train = nothing
test = nothing

open("train.txt", "r") do f
        global train = *((Char).(read(f))...)
end

open("test.txt", "r") do f
        global test = *((Char).(read(f))...)
end

trn_words = split(train, "\r\n")
tst_words = split(test, "\r\n")

for i in range(1;stop=length(trn_words))
        if(trn_words[i] == "")
                continue
        end
        w = lowercase(split(trn_words[i], " ")[1])
        if haskey(word_cnt_dict, w)
                word_cnt_dict[w] += 1
        else
                word_cnt_dict[w] = 1
        end
end

word_to_num = 4
label_to_num = 1

trn_mat_x = []
trn_mat_y = []
cur_len = 0
max_seq_len = -1
cur_seq_x = []
cur_seq_y = []
stop_word=[",", ":", ";", ".", ""]

for word in trn_words
        global word_to_num
        global label_to_num
        global cur_seq_x
        global cur_seq_y
        global stop_word
        global cur_len
        global max_seq_len
        global trn_mat_x
        global trn_mat_y
        word = split(word, " ")
        if(word[1] in stop_word && cur_seq_x != [])
                push!(cur_seq_x, 1)
                push!(cur_seq_y, label_dict["O"])
                cur_len += 1
                if(cur_len > max_seq_len)
                        max_seq_len = cur_len
                end
                push!(trn_mat_x, cur_seq_x)
                push!(trn_mat_y, cur_seq_y)
                cur_seq_x = []
                cur_seq_y = []
                cur_len = 0
                continue
        elseif word[1] == ""
                continue
        end

        if word[1] == "-DOCSTART-"
                continue
        end

        word[1] = lowercase(word[1])

        if word_cnt_dict[word[1]] > min_word_cnt
                if !haskey(word_dict, word[1])
                        word_dict[word[1]] = word_to_num
                        word_to_num += 1
                end
        end

        if !haskey(label_dict, word[end])
                label_dict[word[end]] = label_to_num
                label_to_num += 1
        end

        if haskey(word_dict, word[1])
                push!(cur_seq_x, word_dict[word[1]])
        else
                push!(cur_seq_x, 3)
        end
        push!(cur_seq_y, label_dict[word[end]])
        cur_len += 1
end


test_mat_x = []
test_mat_y = []
cur_len = 0
cur_seq_x = []
cur_seq_y = []
stop_word=[",", ":", ";", ".", ""]

for word in tst_words
        global word_to_num
        global label_to_num
        global cur_seq_x
        global cur_seq_y
        global stop_word
        global cur_len
        global max_seq_len
        global test_mat_x
        global test_mat_y
        word = split(word, " ")
        if(word[1] in stop_word && cur_seq_x != [])
                push!(cur_seq_x, 1)
                push!(cur_seq_y, label_dict["O"])
                cur_len += 1
                if(cur_len > max_seq_len)
                        max_seq_len = cur_len
                end
                push!(test_mat_x, cur_seq_x)
                push!(test_mat_y, cur_seq_y)
                cur_seq_x = []
                cur_seq_y = []
                cur_len = 0
                continue
        elseif word[1] == ""
                continue
        end

        if word[1] == "-DOCSTART-"
                continue
        end

        word[1] = lowercase(word[1])

        if haskey(word_dict, word[1])
                push!(cur_seq_x, word_dict[word[1]])
        else
                push!(cur_seq_x, 3)
        end

        push!(cur_seq_y, label_dict[word[end]])
        cur_len += 1
end

label_dict["null"] = label_to_num

trn_mat_x, trn_mat_y = padding(trn_mat_x, trn_mat_y, label_dict)
test_mat_x, test_mat_y = padding(test_mat_x, test_mat_y, label_dict)

word_dict_str = ""
word_dict_str = word_dict_str * "EOF 1\n"
word_dict_str = word_dict_str * "pad 2\n"
word_dict_str = word_dict_str * "_null_ 3\n"

for key in keys(word_dict)
        global word_dict_str
        word_dict_str = word_dict_str * key * " " * string(word_dict[key]) * "\n"
end

label_dict_str = ""

for key in keys(label_dict)
        global label_dict_str
        label_dict_str = label_dict_str * key * " " * string(label_dict[key]) * "\n"
end

open("word_dict_julia.txt", "w") do f
        global word_dict_str
        write(f, word_dict_str)
end

open("label_dict_julia.txt", "w") do f
        global label_dict_str
        write(f, label_dict_str)
end

using BSON:@save
@save "trn_x.bson" trn_mat_x
@save "trn_y.bson" trn_mat_y
@save "test_x.bson" test_mat_x
@save "test_y.bson" test_mat_y
