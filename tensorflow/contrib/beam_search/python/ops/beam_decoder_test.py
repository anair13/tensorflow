import tensorflow as tf
import numpy as np
from beam_decoder import BeamDecoder

class MarkovChainCell(tf.nn.rnn_cell.RNNCell):
    """
    This cell type is only used for testing the beam decoder.
    It represents a Markov chain characterized by a probability table p(x_t|x_{t-1},x_{t-2}).
    """
    def __init__(self, table):
        """
        table[a,b,c] = p(x_t=c|x_{t-1}=b,x_{t-2}=a)
        """
        assert len(table.shape) == 3 and table.shape[0] == table.shape[1] == table.shape[2]
        self.log_table = tf.log(np.asarray(table, dtype=np.float32))
        self._output_size = table.shape[0]

    def __call__(self, inputs, state, scope=None):
        """
        inputs: [batch_size, 1] int tensor
        state: [batch_size, 1] int tensor
        """
        logits = tf.reshape(self.log_table, [-1, self.output_size])
        indices = state[0] * self.output_size + inputs
        return tf.gather(logits, tf.reshape(indices, [-1])), (inputs,)

    @property
    def state_size(self):
        return (1,)

    @property
    def output_size(self):
        return self._output_size

class RNNTest(tf.test.TestCase):

    def test_markov_outputs(self):
        table = np.array([[[0.9, 0.1, 0, 0],
               [0, 0.9, 0.1, 0],
               [0, 0, 1.0, 0],
               [0, 0, 0, 1.0]]] * 4)

        cell = MarkovChainCell(table)
        initial_state = cell.zero_state(1, tf.int32)
        initial_input = initial_state[0]

        MAX_LEN = 3
        beam_decoder = BeamDecoder(num_classes=4, stop_token=2, beam_size=10, max_len=MAX_LEN)

        outputs, final_state = tf.nn.seq2seq.rnn_decoder(
            [beam_decoder.wrap_input(initial_input)] + [None] * (MAX_LEN-1),
            beam_decoder.wrap_state(initial_state),
            beam_decoder.wrap_cell(cell),
            loop_function = lambda prev_symbol, i: tf.reshape(prev_symbol, [-1, 1])
        )

        with self.test_session(use_gpu=False) as sess:
            beams, probs = sess.run([final_state[2], final_state[3]])

            TRUTH = [0.729, 0.081, 0.081, 0.081, 0.01, 0.009, 0.009, 0.0, 0.0, 0.0] # Calculated by hand
            self.assertAllClose(np.array(TRUTH), np.exp(probs))

if __name__ == "__main__":
    tf.test.main()
