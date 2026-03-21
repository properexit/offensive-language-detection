# Offensive Language Detection (SemEval 2020) -- English and Arabic

This project implements offensive language detection using transformer
models, classical baselines, and cross-lingual transfer.

We evaluated: - English (Tasks A, B, C) - Arabic (Task A) - Zero-shot and
few-shot transfer - Parameter-efficient fine-tuning (LoRA, Freeze)

------------------------------------------------------------------------

## Task Description

-   **Task A (Offensive Language Identification)**\
    Classify whether a tweet is offensive (OFF) or not (NOT)

-   **Task B (Categorization of Offensive Language)**\
    Classify offense type (targeted vs untargeted)

-   **Task C (Offense Target Identification)**\
    Identify target (individual, group, other)

------------------------------------------------------------------------

## 1. Setup

### Clone repository

``` bash
git clone https://gitup.uni-potsdam.de/kale/offensive-language-detection.git
cd offensive-language-detection
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 2. Docker (Optional)

### Pull prebuilt image

``` bash
docker pull properexit/offensive-project
```

### Run

``` bash
docker run properexit/offensive-project
```

### With volume

``` bash
docker run -v $(pwd):/app properexit/offensive-project
```

------------------------------------------------------------------------

## 3. Dataset Setup

Datasets are not included due to large size.

Download from:
https://drive.google.com/file/d/1oMTVEHNvFo15p8jJKIdU6ShgOVOdVlPr/view?ts=69beea56

Place inside root:

    data/raw/
    ├── english/
    ├── arabic/

------------------------------------------------------------------------

## 4. Run Experiments

### Baselines

``` bash
python -m training.train_baseline
```

------------------------------------------------------------------------

### English - Task A

``` bash
python main.py --lang english --task A --config config/english.yaml
```

Without class weights:

    # edit config/english.yaml
    class_weighted: false

------------------------------------------------------------------------

### English - Task B

``` bash
python main.py --lang english --task B --config config/english.yaml
```

------------------------------------------------------------------------

### English - Task C

``` bash
python main.py --lang english --task C --config config/english.yaml
```

------------------------------------------------------------------------

### Multitask (A + B)

``` bash
python main.py --multitask
```

------------------------------------------------------------------------

### PEFT Experiments (English)

LoRA:

``` bash
python main.py --lang english --task A --peft lora --config config/english.yaml
```

Freeze:

``` bash
python main.py --lang english --task A --peft freeze --config config/english.yaml
```

------------------------------------------------------------------------

## 5. Arabic Experiments

### Zero-shot

``` bash
python main.py --lang arabic --task A --mode zero-shot --config config/arabic.yaml
```

------------------------------------------------------------------------

### Few-shot

``` bash
python main.py --lang arabic --task A --mode few-shot --k 1000 --config config/arabic.yaml
```

Tried different values of k: - 1000 - 2000 - 3000

------------------------------------------------------------------------

### Arabic + PEFT

LoRA:

``` bash
python main.py --lang arabic --task A --mode few-shot --k 1000 --peft lora --config config/arabic.yaml
```

Freeze:

``` bash
python main.py --lang arabic --task A --mode few-shot --k 1000 --peft freeze --config config/arabic.yaml
```

------------------------------------------------------------------------

## 6. Error Analysis

``` bash
python -m analysis.error_analysis
```

Outputs: - True Positive - True Negative - False Positive - False
Negative

with confidence scores

------------------------------------------------------------------------

## 7. Checkpoints

Models are saved automatically in:

    checkpoints/

Notes: - Checkpoints are not included in the repository - They are
generated during training

------------------------------------------------------------------------

## 8. Results Summary

### Baselines

-   Majority: 0.40 Macro F1
-   TF-IDF + Logistic Regression: 0.65 Macro F1

------------------------------------------------------------------------

### English Task A

-   Transformer: 0.73 Macro F1
-   Without class weights: 0.734 Macro F1

------------------------------------------------------------------------

### English Task B

-   0.56 Macro F1 (class imbalance issue)

------------------------------------------------------------------------

### English Task C

-   0.44 Macro F1 (multi class difficulty)

------------------------------------------------------------------------

### Multitask

-   0.41 Macro F1 (not effective)

------------------------------------------------------------------------

### Arabic Task A

Zero-shot: - 0.44 Macro F1

Few-shot: - k=1000 -\> 0.78 - k=2000 -\> 0.81 - k=3000 -\> 0.85

------------------------------------------------------------------------

### PEFT

  Method             Performance
  ------------------ -------------
  Full fine-tuning   Best
  LoRA               Moderate
  Freeze             Lowest

------------------------------------------------------------------------

## 9. Key Observations

-   Transformer models outperform classical baselines
-   Class weighting is not always beneficial
-   Multitask learning did not improve performance
-   Cross-lingual transfer is effective for Arabic
-   Performance improves significantly with more Arabic data
-   PEFT methods trade performance for efficiency

------------------------------------------------------------------------

## 10. Environment

Tested with: - Python 3.10 - PyTorch - HuggingFace Transformers

------------------------------------------------------------------------

## 11. Authors

-   Jayesh Choudhari
-   Ayushi Garachh
-   Uday Kale
