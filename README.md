This is the code repository for [Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare](https://arxiv.org/abs/2502.13319)

## Setting up environment

```conda env create -f environment.yml```

## Code

### Vignette

1. To generate rewrite score plot, run ``python src/vignette/get_patching_scores.py``. For example, to generate patching scores for flipping gender to male for the condition multiple sclerosis, using model ``allenai/OLMo-7B-0724-Instruct-hf``, run

```python src/vignette/get_patching_scores.py -demographic_type "gender" -condition "multiple sclerosis" -target "Male" -model_name "allenai/OLMo-7B-0724-Instruct-hf" -output_dir <directory_path>```

2. To generate interchange accuracy, run ``python src/vignette/get_interchange_accuracy.py``. For example, to generate interchange accuracy at layer 18 and window size 0, for flipping gender to male for the condition multiple sclerosis, using model ``allenai/OLMo-7B-0724-Instruct-hf``, run

```python src/vignette/get_interchange_accuracy.py -demographic_type "gender" -condition "multiple sclerosis" -target "Male" -model_name "allenai/OLMo-7B-0724-Instruct-hf" -layer 18 -window 0 -output_dir <directory_path>```

This will give interchange accuracies for scaling factor 1, 2, and 5. You can change this in the file.

Replace ``<directory_path>`` with output directory.

### Depression Risk

1. You will first need to generate brief hospital courses by completing only the first step (Process the MIMIC-IV Summaries) in the preprocessing pipeline [here](https://github.com/stefanhgm/patient_summaries_with_llms/tree/main/preprocess). The csv generated will be your input to the code below.
2. You will also need ``patients.csv.gz`` from the [MIMIC-IV Physionet repo](https://physionet.org/content/mimiciv/3.1/hosp/).

```python src/clinical_applications/preprocess_depression_bhcs.py -summaries_path <summaries_csv> -patients_path <patients_csv_gz> --output_dir <directory_path> -model_type <model_type>```

The output generated can these be used to perform patching experiments. For example, to experiment with gender patching, run 

```python src/clinical_applications/patch_depression.py -batch_index 0 -batch_size 300  -demographic_type "gender" -prompt_index 0 -scale_factor 2 -model_name "allenai/OLMo-7B-0724-Instruct-hf" -bhcs_path <bhcs_path> -output_dir <directory_path> ```

The batching simply allows you to run this code for a subset of BHCs by batching them. The above runs it for the first 300 BHCs. ``batch_index`` of 1 will run it for the next 300 and so on.
``prompt_index`` speicifies which of the 4 prompts to use.  ``scale_factor`` is activation scaling factor.

Note: The code for postprocessing to convert the output to binary Yes/No is very ``allenai/OLMo-7B-0724-Instruct-hf`` specific. We strongly recommend revisiting this code for other models. Revising prompts to ensure strict output format following is also an option.

### Healer cases
To perform patching experiments for Healer cases, for example for race, run

```python src/clinical_applications/patch_healer_cases.py -demographic_type "race" -model_name "allenai/OLMo-7B-0724-Instruct-hf" -output_dir ``<directory_path>``.

In both ``preprocess_depression_bhcs.py`` and ``patch_healer_cases.py``, the scaling factors, token patching positions are set for ``allenai/OLMo-7B-0724-Instruct-hf``. 
