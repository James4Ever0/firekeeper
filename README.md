# FireKeeper

When there is smoke, do something right.

## Usage

First you would build your dataset by recording audio with `record_audio.py`, then rename the recording according to the content.

Make sure the name contains "beep" if the audio file contains the beeping sound from a smoke detector, otherwise not.

Move these files under the folder `dataset`, then run `train_classifier.py`. If the accuracy threshold is satisfied, you will have the model saved.

Run the script `main.py`, as long as there is any potential fire threat detectable by your fellow smoke detectors.
