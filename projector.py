import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector

def split_word_and_vector(filepath):
    word_dict = {}
    with open(filepath, 'r') as f:
        num, dim = f.readline().split()
        array = []
        for _ in range(int(num)):
            line = f.readline().split()
            word, vector = line[0], line[1:]
            word_dict[word] = num
            array.append(vector)
        array = np.array(array)

    return word_dict, array

TARGET_DIR="/home/mil/tonkou/models/research/adversarial_text/train/sst5/unidir"
emb_data_path = os.path.join(TARGET_DIR, "embedding-3000.w2v")

word_dict, data = split_word_and_vector(emb_data_path)
embedding_var = tf.Variable(data, name="embedding")

# Projector設定
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# メタデータ(CSV)パス
# embedding.metadata_path = os.path.join(CUR_DIR, META)

# Projectorに出力
summary_writer = tf.summary.FileWriter(TARGET_DIR)
projector.visualize_embeddings(summary_writer, config)

# 保存準備
saver = tf.train.Saver()

# セッション実行と保存
sess = tf.Session()
sess.run(tf.global_variables_initializer())
_ = saver.save(sess, os.path.join(TARGET_DIR, "vis_embedding.ckpt"), 1)
