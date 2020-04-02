# Specs2Text

This is an in progress reimplementation of the [Jasper](https://arxiv.org/pdf/1904.03288.pdf) speech-to-text network on Intel optimized
TensorFlow using the included Keras API. 

In order to run this network you will first need to download and extract the appropraite training data. In order to do this, set this master folder
as your working directory and run the bash script, download_librispeech.sh, located in the datagen_utils folder. This will download
the datasets listed in the librispeech.csv file. Currently, only the 100 hour clean training set is listed, with the other sets listed in the
extra.csv file. If you wish to download all of the training files, copy-paste the lines in the extra.csv file into the librispeech.csv file, then
run the download_librispeech.sh script. **Warning:** This method takes a *long* time to complete. It may be faster for you to manually download
the appropriate folders from the [librispeech website](http://www.openslr.org/12/) and extract them using tar -xvf. 

After the files have been downloaded and extracted, you will then need to run the covert_json.sh script also located in the 
datagen_utils folder to convert the files from .flac to .wav and construct a .json file containing pertinent information on the audio files.
The convert_json.sh script may need to be edited in order to select exactly which datasets to convert. It is currently set up to convert
the 100 hour clean training set.

Following the conversion of the .flac files and the creation of the .json file, you can then run the train.py script which will initialize and train
a 5x3 Jasper model on the dataset(s) of your choosing. Should the training set you wish to use is different than the 100 hour clean set from
LibriSpeech speech, you will need to update model.compile() with the appropriate max lengths of computed spectrogram and normalized
sentence.

This project has been developed on Intel [Devcloud](https://software.intel.com/en-us/devcloud) and Intel [NUC](https://www.intel.com/content/www/us/en/products/boards-kits/nuc.html).

Package requirements:
- TensorFlow 1.14
- Scipy
- Numpy
- Matplotlib
- Unidecode
- Inflect

You can install the required packages using pip via the terminal by entering ```console pip install -r requirements.txt``` while in the master folder.

To date, I have not had the compute resources necessary to fully train, test, and debug the network. I am still working on securing the necessary resources.
