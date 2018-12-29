include("preprocess_helper.jl")

train_path = "train.txt"
test_path = "test.txt"

word_dict, label_dict, max_seq_len, stop_word, trn_mat_x, trn_mat_y = construct_trn_data(train_path)
test_mat_x, test_mat_y = construct_test_data(test_path, word_dict, label_dict, stop_word, max_seq_len)
save_data(word_dict, label_dict, trn_mat_x, trn_mat_y, test_mat_x, test_mat_y)
