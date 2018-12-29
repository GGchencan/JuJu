using DataStructures
using BSON:@save
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

        """
        padding
                pad the input array and label to the desired length
                :param arr        the input array in array{array} type
                :param arr_y      the label corresponding to the arr
                :param label_dict a dict stored the key and corresponding label
                :param max_length the desired input length
                :param padwidth   maybe the python author know the meaning......
                :return           the array and corresponding label after padding
        """

        for i in range(1; stop=length(arr))
                diff = max_length - length(arr[i])
                if diff > 0
                        arr[i] = vcat(arr[i]..., [2 for i in range(1; stop=diff)]...)
                        arr_y[i] = vcat(arr_y[i]..., [label_dict["null"] for i in range(1; stop=diff)]...)
                end
        end
        return arr, arr_y
end

function save_data(
                word_dict,
                label_dict,
                trn_mat_x,
                trn_mat_y,
                test_mat_x,
                test_mat_y
                )

        word_dict_str = ""
        word_dict_str = word_dict_str * "EOF 1\n"
        word_dict_str = word_dict_str * "pad 2\n"
        word_dict_str = word_dict_str * "_null_ 3\n"

        for key in keys(word_dict)
                word_dict_str = word_dict_str * key * " " * string(word_dict[key]) * "\n"
        end

        label_dict_str = ""

        for key in keys(label_dict)
                label_dict_str = label_dict_str * key * " " * string(label_dict[key]) * "\n"
        end

        open("word_dict_julia.txt", "w") do f
                write(f, word_dict_str)
        end

        open("label_dict_julia.txt", "w") do f
                write(f, label_dict_str)
        end

        trn_mat_x = join(join.(trn_mat_x, " "), "\n")
        trn_mat_y = join(join.(trn_mat_y, " "), "\n")
        test_mat_x = join(join.(test_mat_x, " "), "\n")
        test_mat_y = join(join.(test_mat_y, " "), "\n")

        open("trn_x.txt", "w") do f
                write(f, trn_mat_x)
        end

        open("trn_y.txt", "w") do f
                write(f, trn_mat_y)
        end

        open("test_x.txt", "w") do f
                write(f, test_mat_x)
        end

        open("test_y.txt", "w") do f
                write(f, test_mat_y)
        end


end

function count_word(trn_words, word_cnt_dict)
        """
        count_word
                :param trn_words      a word list
                :param word_cnt_dict  a dict recorded the count of each word
                :return               a renewed dict recorded the count
        """

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

        return word_cnt_dict
end


function construct_trn_data(
                train_path;
                min_word_cnt=5,
                stop_word=[",", ":", ";", ".", ""]
                )

        word_dict = OrderedDict()
        word_cnt_dict = OrderedDict()
        label_dict = OrderedDict()
        train = ""
        open(train_path, "r") do f
                train = *((Char).(read(f))...)
        end

        trn_words = split(train, "\r\n")

        word_cnt_dict = count_word(trn_words, word_cnt_dict)


        word_to_num = 4
        label_to_num = 1

        trn_mat_x = []
        trn_mat_y = []
        cur_len = 0
        max_seq_len = -1
        cur_seq_x = []
        cur_seq_y = []

        # process the train data
        for word in trn_words

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

        label_dict["null"] = label_to_num

        trn_mat_x, trn_mat_y = padding(trn_mat_x, trn_mat_y, label_dict)

        return word_dict, label_dict, max_seq_len, stop_word, trn_mat_x, trn_mat_y
end


function construct_test_data(
                test_path,
                word_dict,
                label_dict,
                stop_word,
                max_seq_len
                )

        # process the test sentences
        test_mat_x = []
        test_mat_y = []
        cur_len = 0
        cur_seq_x = []
        cur_seq_y = []

        test = ""

        open(test_path, "r") do f
                test = *((Char).(read(f))...)
        end

        tst_words = split(test, "\r\n")

        for word in tst_words
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

        test_mat_x, test_mat_y = padding(test_mat_x, test_mat_y, label_dict)

        return test_mat_x, test_mat_y
end
