import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector
import argparse

def make_vocab(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return dict([(line.strip(), i) for i, line in enumerate(f)])

TARGET_DIR="/data/ishimochi0/tonkou/1_Dataset/4_sst/stanfordSentimentTreebank/adv_processed/vocab.txt"
TRAIN_DIR="/home/mil/tonkou/models/research/adversarial_text/train/sst5/unidir_test"
METADATA_DIR=os.path.join(TRAIN_DIR, "metadata.tsv")
CONFIG_PATH=os.path.join(TRAIN_DIR, "projector_config.pbtxt")

vocab = make_vocab(TARGET_DIR)

def write_metadata(vocab, filepath):
    vocab_list = sorted(vocab.items(), key=lambda v:v[1])
    with open(filepath, 'w') as w:
        header = "Word\tId\n"
        w.write(header)
        for word, word_id in vocab_list:
            w.write("{}\t{}\n".format(word, word_id))

def write_emb_config(filepath):
    string = """ embeddings {
    \ttensor_name: "embedding/embedding"
    \tmetadata_path: "metadata.tsv"
    }
    """
    with open(filepath, 'w') as w:
        w.write(string)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('TRAIN_DIR', required=True)
    parser.add_argument('DATA_DIR', required=True)
    args = parser.parse_args()

    METADATA_DIR = os.path.join(args.TRAIN_DIR, "metadata.tsv")
    CONFIG_PATH = os.path.join(args.DATA_DIR, "projector_config.pbtxt")

    write_metadata(vocab, METADATA_DIR)
    write_emb_config(CONFIG_PATH)
