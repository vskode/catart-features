# CatArt Features — README

Quick instructions to set up a Python 3.11 virtual environment, install required packages, and configure the two main project files: `audio_prep.py` and `generate_catart_files.py`.

## Prerequisites
- Python 3.11 installed and available as `python3.11`

## Setup: create virtual environment and install dependencies
Run these commands from the project root:

```bash
# create and activate venv
python3.11 -m venv .venv
source .venv/bin/activate

pip install bacpipe scikit-maad
```


## Configure audio_prep.py
The code that controls the length of audio segments should use a single global variable so the rest of the code can reference it.

Example change to the top of `audio_prep.py`:

```python
# audio_prep.py

# Global segment length (seconds)
CATART_AUDIO_LENGTH = 5.0  # set to desired length in seconds

# to change what indices to include modify the list SELECTED_COLUMNS

```

## Configure generate_catart_files.py
Specify which models to run and where the audio data is located by setting top-level configuration variables.

Example change to `generate_catart_files.py`:

```python
# generate_catart_files.py

# Path to audio files (directory or root)
DATA_DIR = "/path/to/audio_data"

# List of model identifiers / names you want to run.
# Replace with the actual model names or keys used by your project.
MODELS = ['birdnet', 'perch_bird']

# See all available models on https://github.com/bioacoustic-ai/bacpipe

```

- Use absolute paths or project-relative paths.
- Replace placeholder model names with the exact identifiers your runner expects.

## Running the pipeline
1. Activate your venv:
```bash
source .venv/bin/activate
```
2. Generate model files:
```bash
python generate_catart_files.py
```