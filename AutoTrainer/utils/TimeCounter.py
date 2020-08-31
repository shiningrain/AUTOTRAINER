import keras
import time
import csv
import os
import errno
class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        # self.times[self.counter] = (time.time() - self.epoch_time_start)
        self.times.append(time.time() - self.epoch_time_start)
        # self.counter += 1

    @staticmethod
    def make_sure_path_exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def write_to_csv(self, history, log_file, epoch):
        with open(log_file, 'w+') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f)
            w.writerow(history.keys())
            for i in range(len(history['val_loss'])):
                temp = []
                for lis in history.values():
                    temp.append(lis[i])
                w.writerow(temp)
