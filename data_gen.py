# A greatly stripped down version of the data generator from
# https://github.com/robmsmt/KerasDeepSpeech/blob/master/generator.py

import math
import matplotlib.pyplot as plt

from scipy.io import wavfile
from datagen_utils.spectrogram_func import *
from sklearn.utils import shuffle


class WavAudio(object):
    def __init__(self, audio_path, window_length=0.02, skip_window=0.010,
                 fft_length=None, freq_max=8000):
        fs, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32)
        self.audio = audio
        self.fs = fs
        self.fft_length = fft_length or 2 ** math.ceil(math.log2(self.fs*window_length))
        (p_sgram, p_maxtime, p_maxfreq) = sgram(audio, int(skip_window * fs),
                                                int(window_length * fs), self.fft_length,
                                                fs, freq_max)
        self.specgram = np.asarray(p_sgram)
        self.maxtime = p_maxtime
        self.maxfreq = p_maxfreq

    def plot(self):
        return plt.imshow(np.transpose(np.array(self.specgram)),
                          origin='lower',
                          extent=(0, self.maxtime, 0, self.maxfreq),
                          aspect='auto')


class BatchGen(object):
    def __init__(self, manifest, batch_size=16, shuffling=True):
        self.cur_index = 0
        self.batch_size = batch_size
        self.manifest = manifest
        self.shuffling = shuffling

        audiopath = []
        for i in range(len(self.manifest)):
            audiopath.append(self.manifest[i]['audio_filepath'])
        self.audiopaths = audiopath

        duration = []
        for i in range(len(self.manifest)):
            duration.append(self.manifest[i]['duration'])
        self.durations = duration

        transcript = []
        for i in range(len(self.manifest)):
            transcript.append(manifest[i]['transcript'])
        self.transcripts = transcript

        transcript_len = []
        for i in range(len(self.manifest)):
            transcript_len.append(len(self.manifest[i]['transcript']))
        self.max_transcript_len = max(transcript_len)

        del audiopath
        del duration
        del transcript
        del transcript_len

    def get_batch(self, idx):
        batch_x = self.audiopaths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y_trans = self.transcripts[idx * self.batch_size:(idx + 1) * self.batch_size]

        try:
            assert (len(batch_x) == self.batch_size)
            assert (len(batch_y_trans) == self.batch_size)

        except Exception as e:
            print(e)
            print(batch_x)
            print(batch_y_trans)

        batch_input = self.input_gen(batch_x, batch_y_trans)

        return batch_input

    def next_batch(self):
        while 1:
            assert (self.batch_size <= len(self.audiopaths))
            if (self.cur_index + 1) * self.batch_size >= len(self.audiopaths) - self.batch_size:
                self.cur_index = 0
                if(self.shuffling==True):
                    print("SHUFFLING as reached end of data")
                    self.genshuffle()
            try:
                ret = self.get_batch(self.cur_index)
            except:
                print("data error - this shouldn't happen - try next batch")
                self.cur_index += 1
                ret = self.get_batch(self.cur_index)
            self.cur_index += 1
            yield ret

    def genshuffle(self):
        self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                             self.transcript,
                                                             self.finish)

    def input_gen(self, audio_paths, transcripts, normalize=True):
        #print("audio length: {}. transcription length: {}".format(len(audio_paths), len(transcripts)))
        assert (len(audio_paths) == len(transcripts))
        spec = []
        sentence_lengths = []
        spec_lengths = []
        for i in range(len(audio_paths)):
            audio = WavAudio(audio_paths[i][0])
            spec.append(audio.specgram)
            sentence_lengths.append(len(transcripts[i]))
            spec_lengths.append(len(spec[i]))
        spec = np.asarray(spec)

        input_data = np.zeros([spec.shape[0], 2451, spec[0].shape[1]])
        targets = np.ones([len(transcripts), self.max_transcript_len]) * 28  # 28 for coding blank
        label_length = np.zeros([len(transcripts), 1])
        input_length = np.zeros([spec.shape[0], 1])

        for i in range(len(transcripts)):
            input_spec = spec[i]
            target = np.asarray(transcripts[i])
            input_data[i, :input_spec.shape[0], :] = input_spec[i]
            targets[i, :len(target)] = target
            label_length[i] = int(len(target))
            input_length[i] = int(spec[i].shape[0])

        if normalize:
            input_data_norm = self.norm(input_data)

            inputs = {
                'input': input_data_norm.astype('float32'),
                'the_labels': targets.astype('int16'),
                'input_length': input_length.astype('int16'),
                'label_length': label_length.astype('int16')
            }

            outputs = {'ctc': np.zeros([self.batch_size])}
            
            return (inputs, outputs)
        else:
            inputs = {
                'input': input_data.astype('float32'),
                'the_labels': targets.astype('int16'),
                'input_length': input_length.astype('int16'),
                'label_length': label_length.astype('int16')
            }

            outputs = {'ctc': np.zeros([self.batch_size])}
            
            return (inputs,outputs)

    def norm(self, data_array, eps=1e-14):
        return np.asarray((data_array - data_array.mean()) / (data_array.std() + eps))
