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

    # Add evaluation metrics (for EVAL mode)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=-1), predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[1])
    eval_metric_ops = {"accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=train_model_spec['loss'], global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=train_model_spec['loss'],
                                          train_op=train_op, eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(mode=mode, loss=train_model_spec['loss'], eval_metric_ops=eval_metric_ops)


def main():
    parser = argparse.ArgumentParser(description='train the model for all model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default='training_model')
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=0)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    model_dir = args.model_dir

    mnist_generator = MnistDataGenerator(args.batch_size, one_hot=True)

    steps_per_epoch = mnist_generator.num_train_sample // (batch_size * args.num_gpus)

    if args.num_gpus > 0:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus)
        config = tf.estimator.RunConfig(train_distribute=strategy)
        classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='model_dir', config=config)
    else:
        classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='model_dir')

    # Train the Model.
    result = classifier.train(input_fn=lambda: mnist_generator.train_iterator_tf_data(), steps=steps_per_epoch)
    print(result)
    print(result['accuracy'])

    # evaluate the Model.
    result = classifier.evaluate(input_fn=lambda: mnist_generator.test_iterator_tf_data())
    print(result)
    print(result['accuracy'])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
