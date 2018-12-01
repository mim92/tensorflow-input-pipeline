import tensorflow as tf
import numpy as np
import random
import threading
from time import sleep


class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, input_shape, num_classes, batch_size, iterator):
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, ] + input_shape)
        self.dataY = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        self.batch_size = batch_size
        # The actual queue of data. The queue contains a vector for
        self.queue = tf.RandomShuffleQueue(shapes=[input_shape, [num_classes]],
                                           dtypes=[tf.float32, tf.float32],
                                           capacity=2000,
                                           min_after_dequeue=1000)
        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.iterator = iterator
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(self.batch_size)
        return images_batch, labels_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY in self.iterator():
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX, self.dataY: dataY})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class MnistDataGenerator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = np.expand_dims(x_train, axis=-1) / 255.
        self.x_test = np.expand_dims(x_test, axis=-1) / 255.

        self.y_train = tf.keras.utils.to_categorical(y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(y_test, 10)
        self.num_train_sample = len(self.x_train)

    def train_iterator(self):
        while True:
            i = random.randint(0, len(self.x_train) - self.batch_size)
            batch_x = self.x_train[i:i + self.batch_size]
            batch_y = self.y_train[i:i + self.batch_size]

            yield batch_x, batch_y

    def test_iterator(self):
        for i in range(0, len(self.x_test), self.batch_size):
            batch_x = self.x_train[i:i + self.batch_size]
            batch_y = self.y_train[i:i + self.batch_size]
            yield batch_x, batch_y

