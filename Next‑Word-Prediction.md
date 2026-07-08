import tensorflow as tf
import numpy as np
import os

path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f"Corpus length: {len(text)} characters")

vocab = sorted(set(text))
vocab_size = len(vocab)
print(f"Unique characters: {vocab_size}")

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

EMBEDDING_DIM = 256
GRU_UNITS = 1024

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.GRU(GRU_UNITS, return_sequences=True, stateful=False, dropout=0.2),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS)

def generate_text(model, start_string, num_generate=500, temperature=1.0):
    input_indices = [char2idx[s] for s in start_string if s in char2idx]
    if not input_indices:
        input_indices = [char2idx[' ']]
    input_indices = tf.constant(input_indices, dtype=tf.int64)
    input_indices = tf.expand_dims(input_indices, 0)

    generated = list(start_string)

    for _ in range(num_generate):
        predictions = model(input_indices)
        predictions = predictions[:, -1, :]
        predictions = predictions / temperature

        sampled_index = tf.random.categorical(predictions, num_samples=1)
        sampled_index = sampled_index.numpy()[0][0]

        predicted_char = idx2char[sampled_index]
        generated.append(predicted_char)

        input_indices = tf.concat([input_indices[:, 1:], [[sampled_index]]], axis=-1)

    return ''.join(generated)

start_text = "ROMEO: "
print("\n--- Generation with Temperature 0.5 (Conservative) ---")
print(generate_text(model, start_text, num_generate=300, temperature=0.5))

print("\n--- Generation with Temperature 1.0 (Balanced) ---")
print(generate_text(model, start_text, num_generate=300, temperature=1.0))

print("\n--- Generation with Temperature 1.5 (Creative/Random) ---")
print(generate_text(model, start_text, num_generate=300, temperature=1.5))
