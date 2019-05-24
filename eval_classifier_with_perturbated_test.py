from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import graphs
import train_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('pretrained_model_dir', None,
                    'Directory path to pretrained model to restore from')


def main(_):
    """Trains LSTM classification model."""
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        model = graphs.get_model()
        train_op, loss, tensors_op_dict, global_step = model.classifier_training()
        train_utils.run_training(
            train_op,
            loss,
            global_step,
            tensors_op_dict,
            variables_to_restore=model.pretrained_variables,
            pretrained_model_dir=FLAGS.pretrained_model_dir)


if __name__ == '__main__':
    tf.app.run()
