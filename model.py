import tensorflow as tf

Flatten = tf.keras.layers.Flatten()


def build_model(inputs):
    x = tf.layers.conv2d(inputs, 4, (3, 3), activation='relu', padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

    x = tf.layers.conv2d(x, 8, (3, 3), activation='relu', padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

    x = tf.layers.conv2d(x, 8, (3, 3), activation='relu', padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.layers.conv2d(x, 8, (3, 3), activation='relu', padding='same')
    x = tf.layers.batch_normalization(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

    x = Flatten(x)

    x = tf.layers.dense(x, 64, activation='relu')
    softmax = tf.layers.dense(x, 10, activation='softmax')
    return softmax


def model_fn(inputs, reuse=False, is_train=True):
    if 'x' not in inputs:
        ValueError('x is nothing')
    if is_train and 'y' not in inputs:
        ValueError('even training mode, y is nothing')

    with tf.variable_scope('model', reuse=reuse):
        softmax = build_model(inputs['x'])
    model_spec = inputs
    model_spec['softmax'] = softmax
    if 'y' in inputs:
        cross_entropy_loss = -tf.reduce_sum(inputs['y'] * tf.log(softmax))
        if is_train:
            train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)
            model_spec['train_op'] = train_op
        correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(inputs['y'], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.summary.scalar('loss', cross_entropy_loss)
        tf.summary.scalar('acc', accuracy)
        summary_op = tf.summary.merge_all()

        model_spec['loss'] = cross_entropy_loss
        model_spec['accuracy'] = accuracy
        model_spec['summary_op'] = summary_op
    return model_spec


def model_fn_multigpu(inputs, reuse=False, is_train=True):
    if 'x' not in inputs:
        ValueError('x is nothing')
    if is_train and 'y' not in inputs:
        ValueError('even training mode, y is nothing')

    with tf.variable_scope('model', reuse=reuse):
        softmax = build_model(inputs['x'])
    model_spec = inputs
    model_spec['softmax'] = softmax
    if 'y' in inputs:
        cross_entropy_loss = -tf.reduce_sum(inputs['y'] * tf.log(softmax))
        correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(inputs['y'], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        model_spec['loss'] = cross_entropy_loss
        model_spec['accuracy'] = accuracy
    return model_spec

