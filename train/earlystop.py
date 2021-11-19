import pandas as pd

import tensorflow as tf
from tensorflow.data import TFRecordDataset
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util

import numpy as np

import os
import glob
import sys

tf.config.experimental.set_visible_devices([], 'GPU')

def my_summary_iterator(path):
    for r in TFRecordDataset(path):
        yield event_pb2.Event.FromString(r.numpy())

def get_eval_losses(summary_dir):
    losses = []
    steps = []
    for entry in os.scandir(summary_dir):
        if entry.is_file():
            filename = entry.name
            path = os.path.join(summary_dir, filename)
            for event in my_summary_iterator(path):
                for value in event.summary.value:
                    if value.tag == 'eval/loss':
                        losses.append(value.simple_value)
                        steps.append(event.step)
    return losses, steps

if __name__ == '__main__':
    dirs = list(glob.glob(sys.argv[1]))

    min_loss_step = []
    for dir in dirs:
        losses, steps = get_eval_losses(dir)
        print(dir,losses)
        step_min = steps[np.argmin(losses)]
        min_loss_step.append(step_min)

    for dir,step_min in zip(dirs,min_loss_step):
        print('{} Min loss step: {}'.format(dir, step_min))
