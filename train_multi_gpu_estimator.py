import tensorflow as tf
import os
import argparse
from tqdm import tqdm
import logging

from model import model_fn_multigpu
from data_generator import MnistDataGenerator, CustomRunner


def cnn_model_fn(features, labels, mode):
    train_inputs = {'x': features, 'y': labels}
    train_model_spec = model_fn_multigpu(train_inputs, is_train=True)

    predictions = {
        "classes": train_model_spec['prediction'],
        "probabilities": train_model_spec['softmax']
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device('/cpu:0'):
            global_steps = tf.train.get_global_step()
        lr = tf.pow(float(1000), -0.5) * tf.minimum(tf.pow(tf.cast(global_steps, tf.float32), -0.5),
                                                    tf.cast(global_steps, tf.float32) * tf.pow(
                                                        float(1000), -1.5))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        train_op = optimizer.minimize(train_model_spec['loss'], global_step=global_steps)
        return tf.estimator.EstimatorSpec(mode=mode, loss=train_model_spec['loss'], train_op=train_op)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # train_op = optimizer.minimize(loss=train_model_spec['loss'], global_step=tf.train.get_global_step())
        # return tf.estimator.EstimatorSpec(mode=mode, loss=train_model_spec['loss'], train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=-1), predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[1])
    eval_metric_ops = {"accuracy": accuracy}

    return tf.estimator.EstimatorSpec(mode=mode, loss=train_model_spec['loss'], eval_metric_ops=eval_metric_ops)


def train_input_fn(batch_size):
    mnist_generator = MnistDataGenerator(batch_size, one_hot=True)
    dataset = tf.data.Dataset.from_generator(generator=mnist_generator.train_iterator,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=([None, 28, 28, 1], [None, 10]))

    return dataset


def train_input_fn_multi_thread(batch_size, num_calls=16):
    mnist_generator = MnistDataGenerator(batch_size, one_hot=True)
    dataset = tf.data.Dataset.from_generator(generator=mnist_generator.train_iterator_one_shot,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=([28, 28, 1], [10]))
    # dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=num_calls).batch(batch_size=batch_size).prefetch(4)
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=num_calls).batch(batch_size=batch_size)
    return dataset


def main():
    parser = argparse.ArgumentParser(description='train the model for all model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default='estimator_model')
    parser.add_argument('--n_threads', type=int, default=12)
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    model_dir = args.model_dir
    n_threads = args.n_threads

    mnist_generator = MnistDataGenerator(args.batch_size, one_hot=True)

    steps_per_epoch = mnist_generator.num_train_sample // (batch_size * args.num_gpus)
    print(steps_per_epoch)

    strategy = None
    if args.num_gpus > 1:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)

    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    session_config=None)
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=model_dir, config=config)

    # Train the Model.
    for e in range(epochs):
        classifier.train(input_fn=lambda: train_input_fn(batch_size), steps=steps_per_epoch)
        # classifier.train(input_fn=lambda: mnist_generator.train_iterator_tf_data(), steps=steps_per_epoch)

    # evaluate the Model.
    result = classifier.evaluate(input_fn=lambda: mnist_generator.test_iterator_tf_data())

    print(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
