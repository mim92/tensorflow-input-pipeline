import tensorflow as tf
import os
import argparse
from tqdm import tqdm
import logging

from model import model_fn_multigpu
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
        _, loss = sess.run([train_model_spec['train_op'], train_model_spec['loss']])
        losses.append(loss)
        t.set_postfix(train_loss=sum(losses) / len(losses))
    return sum(losses) / len(losses)


def eval(sess, valid_model_spec, test_iterator):
    acces, losses = [], []
    for batch_x, batch_y in test_iterator():
        feed_dict = {valid_model_spec['x']: batch_x, valid_model_spec['y']: batch_y}
        loss, acc = sess.run([valid_model_spec['loss'], valid_model_spec['accuracy']], feed_dict)
        losses.append(loss)
        acces.append(acc)
    return sum(losses) / len(losses), sum(acces) / len(acces)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main():
    parser = argparse.ArgumentParser(description='train the model for all model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default='training_model')
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    model_dir = args.model_dir
    input_shape = [28, 28, 1]
    num_classes = 10
    mnist_generator = MnistDataGenerator(args.batch_size)

    with tf.device("/cpu:0"):
        custom_runner = CustomRunner(input_shape, num_classes, args.batch_size, mnist_generator.train_iterator)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-8)

    valid_inputs = {'x': tf.placeholder(tf.float32, [None, ] + input_shape),
                    'y': tf.placeholder(tf.float32, [None, num_classes])}
    valid_model_spec = model_fn_multigpu(valid_inputs, reuse=False, is_train=False)

    gradients = []
    for gpu_index in range(args.num_gpus):
        with tf.device('/gpu:%d' % gpu_index):
            with tf.name_scope('%s_%d' % ("gpu", gpu_index)) as scope:
                images, labels = custom_runner.get_inputs()
                train_inputs = {'x': images, 'y': labels}

                train_model_spec = model_fn_multigpu(train_inputs, reuse=True, is_train=True)

                tf.add_to_collection(tf.GraphKeys.LOSSES, train_model_spec['loss'])
                losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                total_clone_loss = tf.add_n(losses) / args.num_gpus

                # compute clone gradients
                clone_gradients = optimizer.compute_gradients(total_clone_loss)
                gradients.append(clone_gradients)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(gradients)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optimizer.apply_gradients(grads)

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op)
    train_model_spec = {'train_op': train_op,
                        'loss': total_clone_loss
                        }


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
        custom_runner.start_threads(sess, n_threads=args.n_threads)

        if os.path.isdir(save_dir):
            restore_from = tf.train.latest_checkpoint(save_dir)
            begin_at_epoch = int(restore_from.split('-')[-1])
            saver.restore(sess, restore_from)
            epochs += begin_at_epoch

        for epoch in range(begin_at_epoch, epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
            train_loss = train(sess, train_model_spec, steps_per_epoch)
            valid_loss, valid_acc = eval(sess, valid_model_spec, mnist_generator.test_iterator)
            logging.info('train/loss: {:.4f}'.format(train_loss))
            logging.info('valid/acc: {:.4f}, valid/loss: {:.4f}'.format(valid_acc, valid_loss))
            saver.save(sess, save_path, global_step=epoch + 1)


if __name__ == '__main__':
    main()
