"""
this file is used for the costumer to retrain this model with their own data
    the data format: please give the data according to the conll2003 data format
    the arguments: the first argument(must) is the path to the train data
                   the second argument(optional) is the path to the test data
"""

include("../Data/preprocess_helper.jl")

function get_args()
    """
    to get the train and test data via args
    :return: the path of the train and test data
    """

    args = ARGS

    # check whether the input is legal
    if length(args) > 3 || length(args) < 1
        error("please at least give the train data")
    end

    return args
end

args = get_args()
train_path = args[1]
trn_data_result = construct_trn_data(train_path)
word_dict = trn_data_result[1]
label_dict = trn_data_result[2]
max_seq_len = trn_data_result[3]
stop_word = trn_data_result[4]
trn_mat_x = trn_data_result[5]
trn_mat_y = trn_data_result[6]

if(length(args) == 3)
    test_path = args[2]
    test_mat_x, test_mat_y = construct_test_data(test_path, word_dict, label_dict, stop_word, max_seq_len)
    eval_path = args[3]
    eval_mat_x, eval_mat_y = construct_test_data(test_path, word_dict, label_dict, stop_word, max_seq_len)
    save_data(word_dict, label_dict, trn_mat_x, trn_mat_y, test_mat_x, test_mat_y, eval_mat_x, eval_mat_y)
else
    save_data(word_dict, label_dict, trn_mat_x, trn_mat_y, nothing, nothing, nothing, nothing)
end
