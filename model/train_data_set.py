import numpy as np
import torch
import random
from torch.utils.data import Dataset
from util.dict import WordDict
from util.feature import get_feature


class SpeechData(Dataset):
    def __init__(self, data_path, labels_path, set_type, phone_dict, audio_length=3000):
        super(SpeechData, self).__init__()
        self.data_path = data_path
        self.labels_path = labels_path
        self.set_type = set_type
        self.audio_length = audio_length
        self.wav_paths = []
        self.transcripts = []
        self.encoding = []
        self.wordDict = phone_dict
        self.w2i, self.i2w = self.wordDict.get_dict()
        filename = self.labels_path + "/" + self.set_type + "_labels.txt"
        with open(filename, 'r') as f:
            for line in f:
                items = line.rstrip('\n').split('\t')
                self.wav_paths.append(self.data_path + "/" + self.set_type + "/wav/" + items[0])
                self.transcripts.append(items[1].replace("\n", ""))
        self.data_num = len(self.wav_paths)
        for trn in self.transcripts:
            encoding = [self.w2i[word] for word in trn.split(' ')]
            self.encoding.append(encoding)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        labels = self.encoding[index]

        feature = get_feature(wav_path)

        input_length = len(feature)
        label_length = len(labels)
        return feature, labels, input_length, label_length


def ids2words(encoding, phone_dict):
    return [phone_dict[index] for index in encoding]


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch_sorted = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    freq_size = batch_sorted[0][0].shape[1]
    max_audio_length = batch_sorted[0][0].shape[0]
    batch_size = len(batch_sorted)
    # 找出标签最长的
    batch_temp = sorted(batch_sorted, key=lambda sample: len(sample[1]), reverse=True)
    max_label_length = len(batch_temp[0][1])
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype=np.float32)
    labels = np.ones((batch_size, max_label_length), dtype=np.int32) * -1
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[0]
        label_length = len(target)
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length, :] = tensor[:, :]
        labels[x, :label_length] = target[:]
        input_lens.append(seq_length)
        label_lens.append(label_length)
    input_lens = np.array(input_lens, dtype=np.int64)
    label_lens = np.array(label_lens, dtype=np.int64)
    # 打乱数据
    indices = np.arange(batch_size).tolist()
    random.shuffle(indices)
    inputs = inputs[indices]
    labels = labels[indices]
    input_lens = input_lens[indices]
    label_lens = label_lens[indices]
    return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(input_lens), torch.from_numpy(label_lens)


if __name__ == '__main__':
    wordDict = WordDict(["../../data/train_labels.txt", "../../data/test_labels.txt"])
    dataset = SpeechData("../../data", "../../data", "train", wordDict)
    model_dict = wordDict.get_i2w()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    for batch_idx, sample in enumerate(loader):
        input_data, labels_data, input_lengths, label_lengths = sample
        print("input_data shape:\t" + str(input_data.shape))
        # print(ids2words(labels_data[0][0:5].numpy(), model_dict))
        # print("labels_data shape:\t" + str(labels_data.shape))
        print("input_lengths:\t" + str(input_lengths))
        # print("label_lengths:\t" + str(label_lengths))
        # print("transcripts:\t" + str(transcripts[0]))
        break
