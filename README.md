Text Preprocessing Pipeline for Character‑Level Text Generation

This guide walks through the data preparation phase of a character‑level RNN (e.g., a Shakespeare text generator). We transform a raw string of text into batched integer sequences that a neural network can learn from.
🧠 Why This Pipeline?
Challenge	Our Solution
Text is not numeric – neural networks need numbers	Map every unique character to an integer index.
We need to predict the next character at every position	Create input–target pairs where the target is the input shifted by one character.
Training on the whole corpus at once is impossible	Use tf.data.Dataset to stream, batch, and shuffle efficiently.
We need to capture long‑range dependencies	Use fixed‑length sequences (e.g., 100 characters).
📦 Step 1: Build the Character Mappings

We extract the unique characters from the text and create two lookup structures.
python

vocab = sorted(set(text))                      # e.g., ['a', 'b', 'c', …, 'z']
char2idx = {u: i for i, u in enumerate(vocab)} # char → integer
idx2char = np.array(vocab)                     # integer → char (reverse lookup)

Example with "abcdefghijklmnopqrstuvwxyz":
python

char2idx = {
    'a': 0,  'b': 1,  'c': 2,  'd': 3,  'e': 4,  'f': 5,  'g': 6,  'h': 7,
    'i': 8,  'j': 9,  'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15,
    'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
    'y': 24, 'z': 25
}
idx2char = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                     'y', 'z'])

🔢 Step 2: Convert the Entire Text to Integers

Apply the char2idx mapping to every character in the corpus.
python

text_as_int = np.array([char2idx[c] for c in text])

Input: "abcdefghijklmnopqrstuvwxyz"
Output:
python

array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

    Shape: (26,) – one integer per character.

    The integer 0 represents 'a', 1 represents 'b', and so on.

🔪 Step 3: Create a Dataset of Individual Characters

tf.data.Dataset.from_tensor_slices turns the integer array into a stream of individual tensors.
python

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

If iterated, it yields:
text

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25

Each element is a scalar tf.Tensor of shape () – perfect for batching.
📐 Step 4: Create Sequences of Length SEQ_LENGTH + 1

We batch the character stream into non‑overlapping chunks. Each chunk will become a training sample after splitting.
python

SEQ_LENGTH = 5                     # using 5 for this demo
sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    Each chunk is of length SEQ_LENGTH + 1 = 6.

    drop_remainder=True discards any incomplete final chunk.

For our 26‑character alphabet, we get exactly 4 full chunks (24 characters); the last two characters (y, z) are dropped.
Chunk	Content (integers)	Content (letters)
1	[0, 1, 2, 3, 4, 5]	"abcdef"
2	[6, 7, 8, 9, 10, 11]	"ghijkl"
3	[12, 13, 14, 15, 16, 17]	"mnopqr"
4	[18, 19, 20, 21, 22, 23]	"stuvwx"
✂️ Step 5: Split Each Sequence into Input and Target

This is the core trick – we turn each chunk into a supervised learning pair:

    Input = all characters except the last.

    Target = all characters except the first.

python

def split_input_target(chunk):
    input_text = chunk[:-1]   # all but last
    target_text = chunk[1:]   # all but first
    return input_text, target_text

dataset = sequences.map(split_input_target)

Why? At every timestep t, the model sees character t and must predict character t+1.
For the first chunk [0,1,2,3,4,5]:
Chunk	Input (first 5)	Target (last 5)
Integers	[0,1,2,3,4]	[1,2,3,4,5]
Letters	"abcde"	"bcdef"

After mapping all four chunks:
Sample	Input (int)	Input (letters)	Target (int)	Target (letters)
1	[0,1,2,3,4]	"abcde"	[1,2,3,4,5]	"bcdef"
2	[6,7,8,9,10]	"ghijk"	[7,8,9,10,11]	"hijkl"
3	[12,13,14,15,16]	"mnopq"	[13,14,15,16,17]	"nopqr"
4	[18,19,20,21,22]	"stuvw"	[19,20,21,22,23]	"tuvwx"
🔀 Step 6: Shuffle, Batch, and Prefetch

Finally, we prepare the dataset for efficient model training.
python

BATCH_SIZE = 2   # small for this demo
BUFFER_SIZE = 10000

dataset = (dataset
           .shuffle(BUFFER_SIZE)
           .batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.AUTOTUNE))

Operation	Effect
.shuffle()	Randomises the order of samples to break correlations.
.batch()	Groups samples into mini‑batches of fixed size.
.prefetch()	Overlaps data loading with model execution to eliminate I/O bottlenecks.

With BATCH_SIZE = 2, our 4 samples become 2 batches:

Batch 1 (e.g., samples 3 and 1):
python

# Inputs  shape: (2, 5)
[[12, 13, 14, 15, 16],   # "mnopq"
 [ 0,  1,  2,  3,  4]]   # "abcde"

# Targets shape: (2, 5)
[[13, 14, 15, 16, 17],   # "nopqr"
 [ 1,  2,  3,  4,  5]]   # "bcdef"

Batch 2 (e.g., samples 4 and 2):
python

# Inputs  shape: (2, 5)
[[18, 19, 20, 21, 22],   # "stuvw"
 [ 6,  7,  8,  9, 10]]   # "ghijk"

# Targets shape: (2, 5)
[[19, 20, 21, 22, 23],   # "tuvwx"
 [ 7,  8,  9, 10, 11]]   # "hijkl"

✅ Summary: From Text to Training Batches
Stage	Data Representation	Shape / Type
Raw text	"abcdefghijklmnopqrstuvwxyz"	str
Character mapping	{'a':0, 'b':1, …}	dict
Integerised text	array([0,1,2,…,25])	(26,) np.ndarray
Individual characters	Stream of scalars: 0,1,2,…	Dataset of ()‑shaped tensors
Sequences (chunks)	[0,1,2,3,4,5], [6,7,8,9,10,11], …	Dataset of (6,) tensors
Input–target pairs	(input, target) each of length SEQ_LENGTH	Dataset of ( (5,), (5,) ) tuples
Batches	(input_batch, target_batch) both shape (BATCH_SIZE, SEQ_LENGTH)	(2,5) tensors
💡 Key Insights

    Non‑overlapping chunks – We don’t slide a window over the text; we take contiguous blocks. This still gives the model plenty of local context (e.g., 100 characters) and avoids excessive redundancy.

    One‑shift supervised pairs – By shifting the target by one character, we teach the model to predict the next character at every position, maximising the learning signal.

    tf.data pipeline – Handles shuffling, batching, and prefetching efficiently, making it ready for large corpora.

    Drop remainder – Ensures every batch has exactly BATCH_SIZE samples of length SEQ_LENGTH – crucial for batch‑compatible operations.

This preprocessing pipeline is the foundation for training a character‑level RNN (like the Shakespeare text generator). Once the dataset is ready, you can feed it directly into a model with an Embedding + GRU/LSTM architecture for next‑character prediction.
