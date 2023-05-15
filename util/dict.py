from collections import defaultdict


class WordDict:
    def __init__(self, path_list):
        self.paths = path_list
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.i2w = defaultdict()
        unk = self.w2i['<UNK>']
        for path in self.paths:
            with open(path, 'r') as f:
                for line in f:
                    if line:
                        line = line.split('\t')
                        trns = line[1].replace("\n", "")
                        for pny in trns.split(' '):
                            _ = self.w2i[pny]
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.w2i = defaultdict(lambda: unk, self.w2i)
        self.i2w = defaultdict(lambda: '<UNK>', self.i2w)
        print("字典初始化完成，labels_num：\t" + str(len(self.w2i)))

    def get_dict(self):
        return self.w2i, self.i2w

    def get_i2w(self):
        return self.i2w
