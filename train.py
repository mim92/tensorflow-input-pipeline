import tensorflow as tf
import os
import argparse
from tqdm import tqdm
import logging

from model import model_fn
from data_generator import MnistDataGenerator, CustomRunner


def set_logger(log_file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_file_name))


def train(sess, train_model_spec, steps_per_epoch):
    # Use tqdm for progress bar
    t = tqdm(range(steps_per_epoch))
    acces, losses = [], []
    for _ in t:
        _, loss, acc = sess.run([train_model_spec['train_op'], train_model_spec['loss'], train_model_spec['accuracy']])
        losses.append(loss)
        acces.append(acc)
        t.set_postfix(train_loss=sum(losses) / len(losses), train_acc=sum(acces) / len(acces))
    return sum(losses) / len(losses), sum(acces) / len(acces)


def eval(sess, valid_model_spec, test_iterator):
    acces, losses = [], []
    for batch_x, batch_y in test_iterator():
        feed_dict = {valid_model_spec['x']: batch_x, valid_model_spec['y']: batch_y}
        loss, acc = sess.run([valid_model_spec['loss'], valid_model_spec['accuracy']], feed_dict)
        losses.append(loss)
        acces.append(acc)
    return sum(losses) / len(losses), sum(acces) / len(acces)


def main():
    parser = argparse.ArgumentParser(description='train the model for all model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default='training_model')

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    model_dir = args.model_dir
    input_shape = [28, 28, 1]
    num_classes = 10
    mnist_generator = MnistDataGenerator(args.batch_size)
    custom_runner = CustomRunner(input_shape, num_classes, args.batch_size, mnist_generator.train_iterator)

    images, labels = custom_runner.get_inputs()
    train_inputs = {'x': images, 'y': labels}
    valid_inputs = {'x': tf.placeholder(tf.float32, [None, ] + input_shape),
                    'y': tf.placeholder(tf.float32, [None, num_classes])}

    train_model_spec = model_fn(train_inputs, is_train=True)
    valid_model_spec = model_fn(valid_inputs, reuse=True, is_train=False)

    os.makedirs(model_dir, exist_ok=True)
    set_logger(os.path.join(model_dir, 'train.log'))
    save_dir = os.path.join(model_dir, 'weights')
    save_path = os.path.join(save_dir, 'epoch')
    begin_at_epoch = 0
    steps_per_epoch = mnist_generator.num_train_sample // batch_size

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)  # will keep last 5 epochs
        sess.run(tf.global_variables_initializer())

        tf.train.start_queue_runners(sess=sess)
        custom_runner.start_threads(sess, n_threads=4)

        if os.path.isdir(save_dir):
            restore_from = tf.train.latest_checkpoint(save_dir)
            begin_at_epoch = int(restore_from.split('-')[-1])
            saver.restore(sess, restore_from)
            epochs += begin_at_epoch

        for epoch in range(begin_at_epoch, epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
            train_loss, train_acc = train(sess, train_model_spec, steps_per_epoch)
            valid_loss, valid_acc = eval(sess, valid_model_spec, mnist_generator.test_iterator)
            logging.info('train/acc: {:.4f}, train/loss: {:.4f}'.format(train_acc, train_loss))
            logging.info('valid/acc: {:.4f}, valid/loss: {:.4f}'.format(valid_acc, valid_loss))
            saver.save(sess, save_path, global_step=epoch + 1)


if __name__ == '__main__':
    main()
