from model.train_data_set import WordDict, ids2words
from util.decode import ctc_beam_search_decoder
import torch
from util.feature import get_feature

import time


def predict(model_path, labels_path, wav_path):
    wordDict = WordDict([labels_path + "/train_labels.txt", labels_path + "/test_labels.txt"])
    i2w = wordDict.get_i2w()

    model_input = torch.from_numpy(get_feature(wav_path)).unsqueeze(0)
    input_length = torch.tensor([model_input.shape[1]])

    model = torch.load(model_path)
    model.eval()

    start_time = time.time()

    output, output_lengths = model(model_input, input_length)
    print(output[0].shape)
    output = ctc_beam_search_decoder(output[0].detach().numpy(), 3, i2w)
    # print(output.shape)
    print("out:\t" + str(output))

    end_time = time.time()

    print(end_time - start_time)
