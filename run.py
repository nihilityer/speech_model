import argparse
from predict import predict
from train import train_save

parser = argparse.ArgumentParser(description='基于音节的语音识别模型')

parser.add_argument('--model', required=True, help='模型相关路径')
parser.add_argument('--labels', required=True, help='标签相关路径')
parser.add_argument('--predict', action="store_true", help='是否进行预测')
parser.add_argument('--data', help='数据相关路径')
parser.add_argument('--epoch', type=int, default=100, help='训练次数')
parser.add_argument('--batch', type=int, default=4, help='批次大小')
parser.add_argument('--wav', help='预测音频路径')

# 解析命令行参数
args = parser.parse_args()

if args.predict:
    if args.wav is None:
        raise RuntimeError("请输入需要预测的音频文件路径")
    predict(args.model, args.labels, args.wav)
else:
    if args.data is None:
        raise RuntimeError("训练时需要输入数据路径！")
    train_save(args.data, args.labels, args.model, args.epoch, args.batch)

# python run.py --model ../../model/conformer --labels ../../data --data ../../data

