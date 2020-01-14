import os
from numpy import ceil
from model import small_model
from datagen_utils.manifest import Manifest
from data_gen import BatchGen


labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
          "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
          "v", "w", "x", "y", "z", "'"]

data_dir = os.getcwd() + "/datasets/LibriSpeech"
#data_dir = r"C:\Users\monto\Desktop\IPAsper\LibriSpeech"
#manifest_paths = [r".\LibriSpeech\librispeech-dev-clean-wav.json"]
manifest_paths = [data_dir + "/librispeech-train-clean-100-wav.json"]

manifest = Manifest(data_dir, manifest_paths, labels, len(labels), pad_to_max=False,
                    max_duration=None,
                    sort_by_duration=True,
                    min_duration=None, max_utts=0,
                    normalize=True, speed_perturbation=False)

# TODO extract absolute max lengths for specs and labels for padding purposes
#  Done
train_gen = BatchGen(manifest, batch_size=16)

# compile model with max lengths
model = small_model((None, max(train_gen.spec_lengths), 256),
                    len(manifest.labels_map) + 1, train_gen.max_transcript_len, train=True)


# TODO sequence length error at ctc_3, sequence length(0) <= 760
#  this indicates that the max stepsize of the model is 760
#  the max length of the label should be less than the max step size, where
#  max step size is equal to max input length // 2 (only one instance of stride = 2)
#  SOLVED: reduced strides in first conv layer to 1 so as to not reduce dimensions

# TODO figure out why I am getting increasing loss values
#  SOLVED: dropped the learning rate of SGD from 0.02 to 0.001
model.fit_generator(generator=train_gen.next_batch(),
                    steps_per_epoch=ceil(len(manifest)/16),
                    epochs=20,
                    initial_epoch=0,
                    verbose=1)

model.save_weights('small_model5x3.h5')

# decode_model = IPAsper_model((None, input_data.shape[1], input_data.shape[2]),
#                              len(manifest.labels_map)+1, label_length.max(), train=False)
#
# decode_model.load_weights('10dev.h5')
#
# pred = decode_model.predict(np.expand_dims(input_data[100], axis=0))
#
#
# # Reverse translation of numerical classes back to characters
# def labels_to_text(labs):
#     ret = []
#     for c in labs:
#         if c == len(labels):  # CTC Blank
#             ret.append("_")
#         else:
#             ret.append(labels[c])
#     return "".join(ret)
#
#
# #TODO currently only works with greedy=False
# def decode_predict_ctc(out, top_paths=1):
#     results = []
#     beam_width = 100
#     if beam_width < top_paths:
#         beam_width = top_paths
#     for i in range(top_paths):
#         labs = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
#                                         greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
#         text = labels_to_text(labs)
#         results.append(text)
#     return results
#
#
# decode_predict_ctc(pred, top_paths=5)

