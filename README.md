## DTW Word-Spotting Experiement

This repository contains code that was used for a word-spotting (Query-by-Example) experiment using the DTW technique with MFCC features. The expriment aims to examine the effect of the linguistic practice of Respeaking, described in Reiman's BOLD method, on word-spotting. This experiment is conducted in the context of collaborative transcription for indigenous or endangered languages. The languages used in this particular experiment are Kunwinjku and a simulated low-resource English. The main processing ipynb files are contained in the word_spotting folder. 

## Kunwinjku

Kunwinjku, also known as Gunwinjgu, or Bininj Kunwok to encompass different dialects, is a language spoken in West Arnhem Land in the Northern Territory, Australia. The nominal root -winjku means  "freshwater", so the term Kunwinjku refers to the "freshwater people" of Western Arnhem Land. The language has approximately 1700 speakers in the 2016 census and it is considered a vulnerable language by UNESCO. It is also one of the 13 Indigenous languages that are still being acquired by children in Australia. It is a polysynthetic language that has an exceptionally large and sparse vocabulary.

## Simulated Low-resource English

In the process of constructing the simulated English data, some assumptions were made of the data characteristics in an endangered language scenario. Based on the existing database available for Kunwinjku and other Indigenous Australian Languages on PARADISEC (Pacific And Regional Archive for Digital Sources in Endangered Cultures), an open source database that curates digital materials of small or endangered languages (https://catalog.paradisec.org.au/), we observe the common data characteristics of the speech recordings in an endangered language. 

The speech could often be a monologue or a story told by the elders, a ceremonial speech, an interactive bush-walking guided tour, an interview, a descriptive instruction on procedural knowledge, or a group conversation among speakers. Most speech is produced in an uncontrolled environment with background noise. 

## Speech Recording

Query terms pronunciation and respoken speech are the additional speech recordings tasks that are required for the experiments. The two participants are both young male English speakers who have been learning Kunwinjku. The recording session took place in a quiet office room. The recording was conducted using Audacity, an open-source digital audio recording application, directly on a ASUS laptop. 

Informed by the BOLD methodology, respeaking was conducted using the Lig-Aikuma app in the following steps: (1) load a wav file of spontaneous speech (2) ask participants to listen to the recordings and carefully respeaking correspondingly, sentence by sentence (3) play the respoken recording at the end for error correction.

## Data Processing

As manual segmentation of speech is largely subjective and inconsistent in the respeaking process, the segmentation for the experiment was conducted by applying MAUS to create an Elan transcription file (.eaf) from the text transcription (.txt) and the corresponding audio recording (.wav). The Elan files were checked to ensure correct segmentation. The audio recordings were then segmented into short utterances of sentences based on the sentence-segmented tier of the Elan transcription files. This generated 110 audio files and their corresponding text files in English, and 55 in Kunwinjku.

The pronunciation of query words was recorded in a similar manner, with the speakers reading out the predefined words consecutively for MAUS segmentation of words to take place afterwards. This resulted in the audio files and corresponding text files of the 50 English words and the 45 Kunwinjku words, with two versions pronounced by two different speakers.

## Feature Extraction

The QbE-STD model takes as input raw speech and converts it to frame-level acoustic features using a sliding window feeding into the feature extraction function. The MFCC sequence of frame-level vectors were extracted using the Shennong speech extraction toolbox (https://github.com/bootphon/shennong) that relies on the Kaldi speech recognition technology (https://kaldi-asr.org/) . We extracted 13 MFCC from the audio data and additionally the first and second order derivatives (deltas and delta-deltas), resulting in a 39-dimensional feature vector per 10 ms frame.

## DTW-Cost Calculation

Features were extracted for both the query word exemplar from the word list and the target search utterance in which the word is to be detected. Using the extracted features, the minimum DTW cost of each keywords-utterance pair was calculated and stored as a score to indicate the likelihood of the search target utterance containing the keyword. With an established list of minimum DTW costs of each keywords-utterance pair, a threshold, or cut-off, could then be defined to determine whether the target utterance contain the query word or not. In this research, we set the thresholds at a point that returns maximal F1 score.

In this process, the main DTW-cost calculation script is coded with the Cython implementation based on the code from Herman Kamper's github (https://github.com/kamperh/speech_dtw/blob/master/speech_dtw/_dtw.pyx)

## Evaluation

Performance is reported in terms of both in terms of the area under the curve (AUC) of the receiver operating characteristic (ROC) and best F1-score over the target words.
We assume that linguists have different tolerance level for word predictions in various scenarios. Hence, the DTW-cost threshold of word prediction should be an adjustable feature according the scenario and stages of transcription. Linguists might want to have very few precise correct words automatically transcribed to begin with, then lower the threshold to allow more potential word candidates to be identified.

F1 score is included as an evaluation metric alongside ROC-AUC to function as a more comprehensible and tangible measurement, for the linguists to have a better idea of what the precision and recall could be and whether that performance could be useful. The use of F1 as an evaluation metric is based on the assumption that, on average, word-spotting mechanism of equal emphasis of precision and recall would be most useful and applicable in the language documentation process. This is, however, not to say that there is no potential use case that one measurement (precision/recall) could be more important than another in various scenarios. For example, without sufficient confidence in correctly identifying errors, linguists might prefer to work with word predictions that have a higher precision to begin with, then lower the the threshold to enable more words to be spotted, and double-check the validity of these words manually and view these word spotted results with less confidence. 

