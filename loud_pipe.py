import tensorflow as tf
import numpy as np
import scipy.io.wavefile
import dft

def get_wave_sig(path, b_normalize=True):
    sr, s = wavfile.read(path)
    if b_normalize:
        s = s.astype(np.float32)
        s = (s / np.max(np.abs(s)))
        s -= np.mean(s)
    return s


if __name__ == "__main__":
loud_pipe = "/Users/jeremiesimon/Desktop/06_-_Loud_Pipes.wav"
beat = utils.load_audio("/Users/jeremiesimon/dev/open/kadenze/CADL/session-3/gtzan_music_speech/music_speech/music_wav/beat.wav")

loud_sig = utils.load_audio(loud_pipe)
short_pipe = pipe[:beat.shape[0], 0]

fft_size = 512
hop_size = 256

re, im = dft.dft_np(pipe, hop_size=256, fft_size=512)
mag, phs = dft.ztoc(re, im)
print(mag.shape)
sr = 44100

# We can calculate how many hops there are in a second
# which will tell us how many frames of magnitudes
# we have per second
n_frames_per_second = sr // hop_size

# We want 500 milliseconds of audio in our window
n_frames = n_frames_per_second

# We'll therefore have this many sliding windows:
n_hops = len(mag) // n_frames_per_second

Xs = []
ys = []
for hop_i in range(len(mag)):
    # Creating our window
    frames = mag[(hop_i):(hop_i + 1)]
    log_frame = np.log(np.abs(frames) + 1e-10)
    log_frame = log_frame.reshape(1, 256)
    Xs.append(hop_i)
    ys.append(log_frame)

Xs = np.array(Xs)
X_shape = [len(Xs, 1)]
Xs = Xs[:, np.newaxis]


def loud_model(name,
                input_width,
                output_width,
                n_neurons=30,
                n_layers=10,
                activation_fn=tf.nn.relu,
                final_activation_fn=tf.nn.tanh,
                cost_type='l2_norm'):

    X = tf.placeholder(name='X', shape=[None, input_width],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, output_width],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='{}_layer{}'.format(name, layer_i))[0]

    Y_pred = linear(
        current_input, output_width,
        activation=final_activation_fn,
        name='{}_pred'.format(name))[0]

    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')

    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}

def short_pipe_model(x, name, n_layers = 20, layer_width = 256, activation=tf.nn.tanh):
    Ws = []
    for layer_i in range(n_layers):
        with tf.variable_scope("{}/layer/{}".format(name, layer_i)):
            h, W = utils.linear(x, layer_width, activation=activation)
            Ws.append(W)
            x = h
    return Ws, h

X = tf.placeholder(name='X', shape=[None, 1], dtype=tf.float32)
Ws, Y_pred = build_model(X, "music")

loss = tf.squared_difference(Y_pred, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, reduction_indices=1), reduction_indices=0)
learning_rate = 0.1
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

n_epochs = 200
batch_size = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch_i in range(n_epochs):
    this_accuracy = 0
    iterations = 0
    for Xs_i, ys_i in ds.train.next_batch(batch_size):
        this_accuracy += sess.run([model['cost'], optimizer], feed_dict={model['X']:Xs_i, model['Y']:ys_i})[0]
        iterations += 1
        print("accuracy: ", this_cost)
    print("this_accuracy: ", this_accuracy // iterations)
