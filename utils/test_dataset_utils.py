from dataset_utils import AudioData

clean_audio_path = "./data/MS-SNSD/clean_train" 
noise_audio_path = "./data/MS-SNSD/noise_train"

data : AudioData = AudioData(clean_audio_path=clean_audio_path,
                             noisy_audio_path=noise_audio_path, 
                             sampling_rate=22050,
                             hop_length=512,
                             noise_level=0.1,
                             clean_vec_repeats=1)


