def transformLabel(data_path, out_path):
    labels_file = ["train", "test"]
    for label_type in labels_file:
        out_lines = []
        with open(data_path + "/" + label_type + "/content.txt", "r", encoding='utf8') as f:
            for line in f:
                pair = line.split("\t")
                audio_dir = pair[0][:7]
                out_audio_path = audio_dir + "/" + pair[0]
                seq = pair[1].split(" ")[1::2]
                seq[-1] = seq[-1].replace("\n", "")
                label = ""
                for phone in seq:
                    label += phone + " "
                out_lines.append(out_audio_path + "\t" + label)
        with open(out_path + "/" + label_type + "_labels.txt", "w", encoding='utf8') as f:
            for i in range(len(out_lines)):
                if i != len(out_lines) - 1:
                    f.write(out_lines[i] + "\n")
                else:
                    f.write(out_lines[i])
            f.close()


if __name__ == '__main__':
    transformLabel("../../data", "../data/out")
