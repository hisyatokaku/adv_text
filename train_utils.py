# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for training adversarial text models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import pickle

# Dependency imports

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'Master address.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of parameter servers.')
flags.DEFINE_string('train_dir', '/tmp/text_train',
                    'Directory for logs and checkpoints.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
flags.DEFINE_boolean('log_device_placement', False,
                     'Whether to log device placement.')


def run_training(train_op,
                 loss,
                 global_step,
                 tensors_op_dict=None,
                 variables_to_restore=None,
                 pretrained_model_dir=None):
  """Sets up and runs training loop."""
  tf.gfile.MakeDirs(FLAGS.train_dir)

  # Create pretrain Saver
  if pretrained_model_dir:
    assert variables_to_restore
    tf.logging.info('Will attempt restore from %s: %s', pretrained_model_dir,
                    variables_to_restore)
    saver_for_restore = tf.train.Saver(variables_to_restore)

  # Init ops
  if FLAGS.sync_replicas:
    local_init_op = tf.get_collection('local_init_op')[0]
    ready_for_local_init_op = tf.get_collection('ready_for_local_init_op')[0]
  else:
    local_init_op = tf.train.Supervisor.USE_DEFAULT
    ready_for_local_init_op = tf.train.Supervisor.USE_DEFAULT

  is_chief = FLAGS.task == 0
  sv = tf.train.Supervisor(
      logdir=FLAGS.train_dir,
      is_chief=is_chief,
      save_summaries_secs=30,
      save_model_secs=30,
      local_init_op=local_init_op,
      ready_for_local_init_op=ready_for_local_init_op,
      global_step=global_step)

  # Delay starting standard services to allow possible pretrained model restore.
  with sv.managed_session(
      master=FLAGS.master,
      config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
      start_standard_services=False) as sess:
    # Initialization
    if is_chief:
      if pretrained_model_dir:
        maybe_restore_pretrained_model(sess, saver_for_restore,
                                       pretrained_model_dir)
      if FLAGS.sync_replicas:
        sess.run(tf.get_collection('chief_init_op')[0])
      sv.start_standard_services(sess)

    sv.start_queue_runners(sess)

    # Training loop
    global_step_val = 0
    macro_vars_dict = None

    while not sv.should_stop() and global_step_val < FLAGS.max_steps:
      global_step_val, macro_vars_dict = train_step(sess, train_op, loss, tensors_op_dict, macro_vars_dict, 200, global_step)

    # Final checkpoint
    if is_chief and global_step_val >= FLAGS.max_steps:
      sv.saver.save(sess, sv.save_path, global_step=global_step)

def maybe_restore_pretrained_model(sess, saver_for_restore, model_dir):
  """Restores pretrained model if there is no ckpt model."""
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  checkpoint_exists = ckpt and ckpt.model_checkpoint_path
  if checkpoint_exists:
    tf.logging.info('Checkpoint exists in FLAGS.train_dir; skipping '
                    'pretraining restore')
    return

  pretrain_ckpt = tf.train.get_checkpoint_state(model_dir)
  if not (pretrain_ckpt and pretrain_ckpt.model_checkpoint_path):
    raise ValueError(
        'Asked to restore model from %s but no checkpoint found.' % model_dir)
  saver_for_restore.restore(sess, pretrain_ckpt.model_checkpoint_path)


def train_step(sess, train_op, loss, tensors_op_dict, macro_vars_dict, max_to_keep, global_step):
  """Runs a single training step."""
  start_time = time.time()
  if tensors_op_dict:
      tokens = tensors_op_dict['inputs']
      embedding = tensors_op_dict['cl_embedded']
      embedding_weight = tensors_op_dict['embedding_weight']
      logits = tensors_op_dict['cl_logits']
      perturb = tensors_op_dict['perturb']
      labels = tensors_op_dict['labels']

      _, loss_val, tokens_val, embedding_val, embedding_weight_val, logits_val, perturb_val, labels_val, global_step_val = sess.run(
          [train_op, loss, tokens, embedding, embedding_weight, logits, perturb, labels, global_step])
  else:
      _, loss_val, global_step_val = sess.run([train_op, loss, global_step])
  duration = time.time() - start_time

  vars_dict = (
      {'embedding': [embedding_val],
       'perturb': [perturb_val],
       'labels': [labels_val],
       'tokens': [tokens_val],
       'embedding_weight': [embedding_weight_val],
       'logits': [logits_val],
       'global_step': [global_step_val],
       }
  )

  # update macro_vars_dict, if its 0th step, create macro_vars_dict
  if macro_vars_dict is None:
    macro_vars_dict = vars_dict
  else:
    for k, v in vars_dict.items():
      macro_vars_dict[k] += v
  if len(macro_vars_dict['global_step']) % max_to_keep == 0:
    for k, v in vars_dict.items():
      # discard old batch
      macro_vars_dict[k] = macro_vars_dict[k][-max_to_keep:]
  # Logging
  if global_step_val % 100 == 0:
    examples_per_sec = FLAGS.batch_size / duration
    sec_per_batch = float(duration)

    format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
    tf.logging.info(format_str % (global_step_val, loss_val, examples_per_sec,
                                  sec_per_batch))

  if tensors_op_dict and global_step_val % 1000 == 0:
    with open(os.path.join(FLAGS.train_dir, "vars-{}.pkl".format(global_step_val)), 'wb') as f:
        pickle.dump(macro_vars_dict, f)

  if np.isnan(loss_val):
    raise OverflowError('Loss is nan')

  return global_step_val, macro_vars_dict

def compute_intermediate_tensors_step(sess, tensors_op_dict, global_step):
  keys = [k for k, v in tensors_op_dict.items()]
  ops = [v for k, v in tensors_op_dict.items()]
  sess.run(ops)
  pass