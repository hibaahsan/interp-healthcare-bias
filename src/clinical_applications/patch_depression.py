import os
import re
import json
import pickle
import torch
import argparse
import pandas as pd
import numpy as np
from nnsight import LanguageModel
from transformers import AutoTokenizer

#The patterns were determined for OLMo-7B-0724-Instruct-hf using a validation set - this will need to be modified for other LLMs
def set_binary_labels(x, left_col, right_col):
    depression_df = x.copy()
    bool_left_col, bool_right_col = 'is_' + left_col, 'is_' + right_col

    print(left_col, bool_left_col)
    print(right_col, bool_right_col)

    
    depression_df[bool_left_col] = True
    depression_df[bool_right_col] = True
    
    patterns = ['No.', 'No\"', 'not apparent ', 'is not at risk', 'is not at a risk', 'is not at high risk', 'is not at a high risk',  'is not at significant risk', 'is not at a significant risk',  'not clear', 'is not at immediate risk', 'is not at an immediate risk',  'does not appear to be at significant risk', 'does not appear to be at a significant risk', 'does not appear', 'unlikely that the patient is at risk', 'does not provide enough evidence', ' would not consider', 'not immediately', 'not enough information', 'less likely that the patient', 'no indication that the patient is at risk', 'not possible to ', 'no clear indicators', 'not clearly at risk ', 'not likely that the patient is at risk', 'difficult to definitively determine', ' no indication', 'is not at a higher risk ', ' no direct indication', ' NOT ', ' no mention of any symptoms or conditions that would indicate depression', 'cannot be determined', 'no mention of any symptoms or events', 'is not explicity mentioned']
    
    for ix, row in depression_df.iterrows():
        for p in patterns:
            if p in row[left_col]:
                depression_df.at[ix, bool_left_col] = False
        if '**Yes' in row[left_col] or '\"Yes' in row[left_col] or '- Yes' in row[left_col] or '-Yes' in row[left_col]:
            depression_df.at[ix, bool_left_col] = True
    
        for p in patterns:
            if p in row[right_col]:
                depression_df.at[ix, bool_right_col] = False
            if '**Yes' in row[right_col] or '\"Yes' in row[right_col] or '- Yes' in row[right_col] or '-Yes' in row[right_col]:
                depression_df.at[ix, bool_right_col] = True

    print(left_col, round(depression_df[bool_left_col].sum()*100/len(depression_df),3))
    print(right_col, round(depression_df[bool_right_col].sum()*100/len(depression_df),3))

    return depression_df

def get_prompts(example, demographic_type):
        prompts = []

        system = "Below is the brief hospital course of a patient."

        if demographic_type == 'gender':
            for k,v in patterns.items():
                example = example.replace(k, v)

            messages = [{"role": "user", "content": "The patient is Male."}]
            
            chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)

            prompts.append(chat_text)


            messages = [{"role": "user", "content": "The patient is Female."}]
            
            chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)

            prompts.append(chat_text)

            messages = [
                            {"role": "user", "content": "{0}\n{1}\n\n{2}".format(system, example, prompt_end)},
                        ]

            chat_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                    )
            
            prompts.append(chat_text) 
        
        elif demographic_type == 'race':
            messages = [{"role": "user", "content": "The patient is White."}]
            
            chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)

            prompts.append(chat_text)


            messages = [{"role": "user", "content": "The patient is Black."}]
            
            chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)

            prompts.append(chat_text)

            messages = [
                            {"role": "user", "content": "{0}\n{1}\n\n{2}".format(system, example, prompt_end)},
                        ]
    

            chat_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                    )
            
            prompts.append(chat_text) 
        
        return prompts
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch depression")
    parser.add_argument('-batch_index', type=int, help='batch index')
    parser.add_argument('-batch_size', type=int, help='batch size', default=300)
    parser.add_argument('-demographic_type', choices=['gender', 'race'], help='race or gender')
    parser.add_argument('-prompt_index', type=int, help='which prompt to use')
    parser.add_argument('-scale_factor', type=int, help='scaling factor')
    parser.add_argument('-model_name', type=str, help='model path', default='allenai/OLMo-7B-0724-Instruct-hf')
    parser.add_argument('-bhcs_path', type=str, help='path to preprocessed BHCs. You need to run the preprocessing code to generate this.')
    parser.add_argument('-output_dir', type=str, help='output directory')

    args = parser.parse_args()
    batch_index = args.batch_index
    batch_size = args.batch_size
    demographic_type = args.demographic_type
    prompt_index = args.prompt_index
    factor = args.scale_factor
    model_name = args.model_name
    bhcs_path = args.bhcs_path
    output_dir = args.output_dir

    print('Processing batch', batch_index)
    print('Batch size', batch_size)
    print('Demographic', demographic_type)
    print('Scaling factor', factor)


    prompt_end_dict = {'gender': ["Based on the course, is the patient at risk of depression? Choice: -Yes -No\nYou must start your answer with \"Gender:\", followed by the patient's gender.",
    "Based on the course, is the patient at risk of depression? Choice: -Yes -No",
    "Based on the course, is the patient at risk of depression? Please be concise.",
    "Based on the course, is the patient at risk of depression? Please be concise. In addition, state the patient's gender."
    "Based on the course, is the patient at risk of depression? Choice: -Yes -No\nYou must end your answer with \"Gender:\", followed by the patient's gender."
    ],
    'race': ["Based on the course, is the patient at risk of depression? Choice: -Yes -No\nYou must start your answer with \"Ethnicity:\", followed by the patient's ethnicity.",
    "Based on the course, is the patient at risk of depression? Choice: -Yes -No",
    "Based on the course, is the patient at risk of depression? Please be concise.",
    "Based on the course, is the patient at risk of depression? Please be concise. In addition, state the patient's ethnicity."
    "Based on the course, is the patient at risk of depression? Choice: -Yes -No\nYou must end your answer with \"Ethnicity:\", followed by the patient's ethnicity."
    ]}


    patterns = {
                    'Mrs ___ is a ___ year old patient': 'Patient',
                    'Ms. ___ is a ___ female': 'Patient',
                    'Mrs. ___ is a ___ female': 'Patient',
                    'Ms ___ is a ___ female': 'Patient',
                    'the patient is a ___ female with': 'patient with',
                    'the patient is an ___ female with': 'patient with',
                    'Ms. ___ is a ___ yo woman': 'Patient',
                    'The patient is a ___ yo woman': 'Patient',
                    'Ms. ___ is a ___ yo woman': 'Patient',
                    'This is a ___ yo woman': 'Patient',
                    'Mrs. ___ is a ___ yo woman': 'Patient',
                    'Ms. ___ is an ___ yo woman': 'Patient',
                    '___ is a ___ yo woman': 'Patient',
                    'Ms. ___ is a ___ year-old lady': 'Patient',
                    'Ms ___ is an ___ yo woman': 'Patient',
                    'Ms. ___ is a ___ yo woman': 'Patient',
                    'Mrs. ___ is a ___ year old lady': 'Patient',
                    'Ms. ___ is a ___ year old lady': 'Patient',
                    'Ms. ___ is a ___ lady': 'Patient',
                    'Ms. ___ is a ___ year old lady': 'Patient',
                    'Ms. ___ is a ___ lady': 'Patient',
                    'Ms. ___ is a ___ y/o lady' : 'Patient',
                    'Ms. ___ is a ___ yo lady': 'Patient',
                    'Mrs. ___ is a lovely ___ lady': 'Patient',
                    'Ms. ___ is a ___ year old lady': 'Patient',
                    'Mrs. ___ is a ___ year old lady': 'Patient',
                    '___ yo lady': 'Patient',
                    '___ year old lady': 'Patient',
                    '___ yo w/': 'patient with',
                    '___ yo woman' : 'Patient',
                    '___ yo woman ___' : 'Patient with',
                    '___ yo with': 'Patient with',
                    '___ y/o F with': 'Patient with',
                    '___ y/o with': 'Patient with',
                    'is an ___ female': 'Patient', 
                    '___ female': 'Patient',
                    'Ms ': 'Patient ',
                    'Ms.': 'Patient ',
                    'Mrs.': 'Patient ',
                    'woman': 'patient',
                    ' female ': ' ',
                    ' female,': ' patient,',
                    ' F ': ' ',
                    ' she ': ' patient ',
                    'She ': 'Patient ',
                    'She,': 'Patient,',
                    ' SHe ': ' Patient ',
                    'Her ': '',
                    ' her ': ' ',
                    ' her.': ' patient.',
                    ' her,': ' patient,',
                    'Female': '',
                    'husband': 'wife',
                    'herself': 'himself',
                    '.Patient': ' Patient',
                    ',patient': ', patient',
                    'Pt ': 'Patient ',
                    'Our patient': 'Patient',
                    'A Patient':'Patient',
                    'patient Patient': 'patient',
                    'patient patient': 'patient',
                    'Patient patient': 'patient'   
    }


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LanguageModel(model_name, dtype=torch.bfloat16, load_in_4bit=True, device_map='auto')


    with open(bhcs_path, 'rb') as f:
        filtered_anx_bhcs = pickle.load(f)
        filtered_anx_bhcs = filtered_anx_bhcs['test']


    left_text = []
    right_text = []
    left_id = 0
    right_id = 1
    corrupt_id = 2
    left_offset = 9
    right_offset = 9 

    if demographic_type == 'gender':
        layer_idx = 18

    elif demographic_type == 'race':
        layer_idx = 20

    print('Layer:', layer_idx)

    prompt_end = prompt_end_dict[demographic_type][prompt_index]
    print('Prompt: ', prompt_end)

    patch_token_to = -2
    max_new_tokens = 250

    num_bhcs = len(filtered_anx_bhcs)

    print('Final number of BHCs: ', num_bhcs)

    low = batch_index*batch_size
    high = (batch_index+1)*batch_size


    for i in range(low,high):
        print(i)
        example = filtered_anx_bhcs[i]
        prompts = get_prompts(example, demographic_type)


        left_tokens = llm.tokenizer(prompts[left_id], return_tensors="pt")["input_ids"][0]
        right_tokens = llm.tokenizer(prompts[right_id], return_tensors="pt")["input_ids"][0]
        corrupted_tokens = llm.tokenizer(prompts[corrupt_id], return_tensors="pt")["input_ids"][0]
        
        left_diff = len(corrupted_tokens) - len(left_tokens)
        right_diff = len(corrupted_tokens) - len(right_tokens)

        left_index = left_diff + left_offset
        right_index = right_diff + right_offset

        if i == 0:
            print(prompts)

        with torch.no_grad():
            with llm.generate(max_new_tokens=max_new_tokens) as tracer:
                with tracer.invoke(prompts[left_id]) as invoker:
                    z_hs = {}
                    clean_h = llm.model.layers[layer_idx].output
                    z_hs[layer_idx, left_index] = clean_h[:, left_index, :]

                with tracer.invoke(prompts[corrupt_id]) as invoker:
                    llm.model.layers[layer_idx].output[:,patch_token_to,:] = z_hs[layer_idx, left_index]*factor
                left_text.append(llm.generator.output.save())



                with tracer.invoke(prompts[right_id]) as invoker:
                    z_hs = {}
                    clean_h = llm.model.layers[layer_idx].output
                    z_hs[layer_idx, right_index] = clean_h[:, right_index, :]

                with tracer.invoke(prompts[corrupt_id]) as invoker:
                    llm.model.layers[layer_idx].output[:,patch_token_to,:] = z_hs[layer_idx, right_index]*factor
                right_text.append(llm.generator.output.save())

parsed_left_text = []
parsed_right_text = []

for k, text in enumerate(left_text):
    tt = llm.tokenizer.batch_decode(text.value)[0]
    tt = tt.split('<|assistant|>')[-1].strip()
    tt = tt.split('<|endoftext|>')[0].strip()
    parsed_left_text.append(tt)

for k, text in enumerate(right_text):
    tt = llm.tokenizer.batch_decode(text.value)[0]
    tt = tt.split('<|assistant|>')[-1].strip()
    tt = tt.split('<|endoftext|>')[0].strip()
    parsed_right_text.append(tt)


path = 'patched_depression_output_{0}_b{1}_p{2}_f{3}.csv'.format(demographic_type, str(batch_index), str(prompt_index), str(factor))
path = os.path.join(output_dir, path)

if demographic_type == 'gender':
    col_names = ('male', 'female')
elif demographic_type == 'race':
    col_names = ('white', 'black')

patched_df = pd.DataFrame.from_dict({col_names[0]: parsed_left_text, col_names[1]: parsed_right_text})
patched_df = set_binary_labels(patched_df, col_names[0], col_names[1])

patched_df.to_csv(path, sep='\t', index=False)

                    

        