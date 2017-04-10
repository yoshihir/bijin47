'''Create model.'''
import os
import random
import numpy as np
import tensorflow as tf

label_dict = {
    'hokkaido':0,
    'aomori':1,
    'iwate':2,
    'sendai':3, # miyagi
    'akita':4,
    #'yamagata':5, nothing
    'fukushima':6,
    'ibaraki':7,
    'tochigi':8,
    'gunma':9,
    'saitama':10,
    'chiba':11,
    'tokyo':12,
    'kanagawa':13,
    'niigata':14,
    #'toyama':15, nothing
    'kanazawa':16,
    'fukui':17,
    'yamanashi':18,
    'nagano':19,
    #'gifu':20, nothing
    'shizuoka':21,
    'nagoya':22, #aichi
    #'mie':23, nothing
    #'shiga':24, nothing
    'kyoto':25,
    'osaka':26,
    'kobe':27, #hyogo
    'nara':28,
    #'wakayama':29, nothing
    'tottori':30,
    #'shimane':31, nothing
    'okayama':32,
    'hiroshima':33,
    'yamaguchi':34,
    'tokushima':35,
    'kagawa':36,
    #'ehime':37, nothing
    #'kochi':38, nothing
    'fukuoka':39,
    'saga':40,
    'nagasaki':41,
    'kumamoto':42,
    #'oita':43, nothing
    'miyazaki':44,
    'kagoshima':45,
    'okinawa':46,
    }

def load_data(data_type):
    filenames, images, labels = [], [], []
    walk = list(filter(lambda _:data_type in _[0], os.walk('faces')))
    for (root, dirs, files) in walk:
        filenames += ['{}/{}'.format(root, _) for _ in files]
    # Shuffle files
    random.shuffle(filenames)
    # Read, resize, and reshape images
    images = list(map(lambda _: tf.image.decode_jpeg(tf.read_file(_), channels=3), filenames))
    images = list(map(lambda _: tf.image.resize_images(_, [32, 32]), images))
    images = list(map(lambda _: tf.reshape(_, [-1]), images))
    for filename in filenames:
        label = np.zeros(47)
        for k, v in label_dict.items():
            if k in filename:
                label[v] = 1.
        labels.append(label)

    return images, labels


def get_batch_list(l, batch_size):
    # [1, 2, 3, 4, 5,...] -> [[1, 2, 3], [4, 5,..]]
    return [np.asarray(l[_:_+batch_size]) for _ in range(0, len(l), batch_size)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(images_placeholder, keep_prob):
    # Convolution layer
    x_image = tf.reshape(images_placeholder, [-1, 32, 32, 3])
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Pooling layer
    h_pool2 = max_pool_2x2(h_conv2)

    # Full connected layer
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Full connected layer
    W_fc2 = weight_variable([1024,47])
    b_fc2 = bias_variable([47])

    return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


def main():
    with tf.Graph().as_default():
        train_images, train_labels = load_data('train')
        test_images, test_labels = load_data('test')
        x = tf.placeholder('float', shape=[None, 32 * 32 * 3])  # 32 * 32, 3 channels
        y_ = tf.placeholder('float', shape=[None, 47])  # 47 classes
        keep_prob = tf.placeholder('float')

        y_conv = inference(x, keep_prob)
        # Loss function
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)
        # Minimize cross entropy by using SGD
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        tf.summary.scalar('accuracy', accuracy)

        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)

        batched_train_images = get_batch_list(train_images, 25)
        batched_train_labels = get_batch_list(train_labels, 25)

        train_images = list(map(lambda _: sess.run(_).astype(np.float32) / 255.0, np.asarray(train_images)))
        test_images = list(map(lambda _: sess.run(_).astype(np.float32) / 255.0, np.asarray(test_images)))
        train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)

        # Train
        for step, (images, labels) in enumerate(zip(batched_train_images, batched_train_labels)):
            images = list(map(lambda _: sess.run(_).astype(np.float32) / 255.0, images))
            sess.run(train_step, feed_dict={ x: images, y_: labels, keep_prob: 0.5 })
            train_accuracy = accuracy.eval(feed_dict = {
                x: train_images, y_: train_labels, keep_prob: 1.0 })
            print ('step {}, training accuracy {}'.format(step, train_accuracy))
            summary_str = sess.run(summary_op, feed_dict={
                x: train_images, y_: train_labels, keep_prob: 1.0 })
            summary_writer.add_summary(summary_str, step)
        # Test trained model
        test_accuracy = accuracy.eval(feed_dict = {
            x: test_images, y_: test_labels, keep_prob: 1.0 })
        print ('test accuracy {}'.format(test_accuracy))
        # Save model
        # Windows
        cwd = os.getcwd()
        save_path = saver.save(sess, cwd + "/model.ckpt")

if __name__ == '__main__':
    main()