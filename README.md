-- STRATEGY -- 
Step 1: Build the Character Mappings
python

vocab = sorted(set(text))  # -> ['a', 'b', 'c', ..., 'z']
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

What char2idx looks like:
python

{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 
 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 
 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

idx2char is the reverse lookup: np.array(['a','b','c',...'z']).
Step 2: Convert the Entire Text to Integers
python

text_as_int = np.array([char2idx[c] for c in text])

Input: "abcdefghijklmnopqrstuvwxyz" (string)
Output:
python

array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

    Shape: (26,)

    Interpretation: The letter 'a' becomes 0, 'b' becomes 1, …, 'z' becomes 25.

Step 3: Create a Dataset of Individual Characters
python

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

What it yields (if iterated):
python

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25

    Each element is a single integer scalar (tf.Tensor of shape ()).

Step 4: Create Sequences of Length SEQ_LENGTH + 1 (i.e., 6)
python

sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

This takes non‑overlapping chunks of 6 characters from the stream.
26 characters ÷ 6 = 4 full chunks (24 characters). The remaining 2 characters (y and z) are dropped because drop_remainder=True.

The 4 chunks (each is a tensor of length 6):
Chunk	Content (integers)	Content (letters)
Chunk 1	[0, 1, 2, 3, 4, 5]	"abcdef"
Chunk 2	[6, 7, 8, 9, 10, 11]	"ghijkl"
Chunk 3	[12, 13, 14, 15, 16, 17]	"mnopqr"
Chunk 4	[18, 19, 20, 21, 22, 23]	"stuvwx"
Step 5: Split Each Sequence into Input and Target
python

def split_input_target(chunk):
    input_text = chunk[:-1]   # all but last
    target_text = chunk[1:]   # all but first
    return input_text, target_text

dataset = sequences.map(split_input_target)

This is the core magic – it shifts the sequence by one character to create the supervised learning pairs.

Transformation for Chunk 1 ([0,1,2,3,4,5]):

    input_text = [0,1,2,3,4] (letters "abcde")

    target_text = [1,2,3,4,5] (letters "bcdef")

Why? At timestep 0, the model sees 'a' and must predict 'b'. At timestep 1, it sees 'b' and predicts 'c', and so on. The loss is calculated across all 5 timesteps simultaneously.

After applying split_input_target to all 4 chunks, the dataset now contains these 4 elements:
Element	Input (integers)	Input (letters)	Target (integers)	Target (letters)
1	[0,1,2,3,4]	"abcde"	[1,2,3,4,5]	"bcdef"
2	[6,7,8,9,10]	"ghijk"	[7,8,9,10,11]	"hijkl"
3	[12,13,14,15,16]	"mnopq"	[13,14,15,16,17]	"nopqr"
4	[18,19,20,21,22]	"stuvw"	[19,20,21,22,23]	"tuvwx"
Step 6: Shuffle, Batch, and Prefetch
python

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    .shuffle(BUFFER_SIZE): Randomly shuffles the 4 elements. Let’s say the order becomes: Element 3, Element 1, Element 4, Element 2.

    .batch(BATCH_SIZE=2, drop_remainder=True): Groups the shuffled elements into batches of 2. Since we have 4 elements, we get 2 batches.

Batch 1 (e.g., contains Elements 3 and 1):

    Inputs tensor shape (2, 5):
    python

[[12, 13, 14, 15, 16],   # "mnopq"
 [ 0,  1,  2,  3,  4]]   # "abcde"

Targets tensor shape (2, 5):
python

[[13, 14, 15, 16, 17],   # "nopqr"
 [ 1,  2,  3,  4,  5]]   # "bcdef"

Batch 2 (contains Elements 4 and 2):

    Inputs tensor shape (2, 5):
    python

[[18, 19, 20, 21, 22],   # "stuvw"
 [ 6,  7,  8,  9, 10]]   # "ghijk"

Targets tensor shape (2, 5):
python

[[19, 20, 21, 22, 23],   # "tuvwx"
 [ 7,  8,  9, 10, 11]]   # "hijkl"

.prefetch(tf.data.AUTOTUNE): While the model is training on Batch 1, the CPU is already preparing Batch 2 in the background, eliminating I/O wait time.
