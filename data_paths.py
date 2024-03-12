import os

data_config = {
        "data_folder_path" : "./data",
        "folder_name" : "MS-SNSD",
        "sample_rate" : 16000,
        "hop_length" : 512,
        "noise_level" : 0.1,
        "frame_length" : 512 * 4,
        "fft_length" : 512 * 4,
        "clean_vec_repeats" : 1
}

folder_path = os.path.join(data_config["data_folder_path"], data_config["folder_name"])

data_paths = {
        "clean" : { 
            "train" : os.path.join(folder_path, "clean_train"),
            "test" : os.path.join(folder_path, "clean_test")
        },
        "noise" : { 
            "train" : os.path.join(folder_path, "noise_train"),
            "test" : os.path.join(folder_path, "noise_test")
        },
        "spectrograms" : {
            "train" : os.path.join(folder_path, "tf_train_spects"),   
            "test" : os.path.join(folder_path, "tf_test_spects")
        },

        "vectors" : {
            "train" : os.path.join(folder_path, "tf_train_vecs"),   
            "test" : os.path.join(folder_path, "tf_test_vecs"),   
        }
}
