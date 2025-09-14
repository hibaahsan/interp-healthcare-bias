import os
import json
import pickle
import torch
import argparse
import pandas as pd
import numpy as np
import nnsight
from nnsight import LanguageModel
from transformers import AutoTokenizer

def is_success(row):
    target = row['target'].lower() 
    text = row['text'].lower()

    if target == 'african american':
        target = 'african'

    pred = ''

    if target in ['male', 'female']:
        if 'woman' in text or 'lady' in text or 'female' in text:
            pred = 'female'
        elif ' man' in text or 'gentleman' in text or 'male' in text:
            pred = 'male'
    elif 'african' in text or ' black' in text:
        pred = 'african'
    elif ' hispanic' in text or ' mexican' in text or 'latin' in text:
        pred = 'hispanic'
    elif  ' asian' in text or ' chinese' in text:
        pred = 'asian'
    elif 'caucasian' in text or 'white' in text:
        pred = 'caucasian'

    return target == pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get interchange accuracy")
    parser.add_argument('-target', type=str, help='Male/Caucasian/Hispanic/Female')
    parser.add_argument('-demographic_type', choices=['gender', 'race'], help='race or gender')
    parser.add_argument('-condition', type=str, help='biased condition')
    parser.add_argument('-layer', type=int, help='layer')
    parser.add_argument('-window', type=int, help='layer window', default=0)
    parser.add_argument('-model_name', choices=['allenai/OLMo-7B-0724-Instruct-hf', 'meta-llama/Llama-3.1-8B-Instruct', 'google/gemma-2-9b-it', 'allenai/OLMo-2-0325-32B-Instruct'], help='model type', default='allenai/OLMo-7B-0724-Instruct-hf')
    parser.add_argument('-output_dir', type=str)


    args = parser.parse_args()
    target = args.target
    demographic_type = args.demographic_type
    source_condition = args.condition
    layer = args.layer
    window = args.window
    model_name = args.model_name
    output_dir = args.output_dir


    target_condition = target[0].upper() + target[1:].lower()
    if target.lower() == 'african american':
        target_condition = 'African American'
        
    print('Target:', target_condition)
    print('Demographic Type:', demographic_type)
    print('Condition:', source_condition)
    print('Layer:', layer)
    print('Window:', window)
    print('Model type:', model_name)


    text_templates = {
        'allenai/OLMo-7B-0724-Instruct-hf': {'role': 'user', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history.", 'gender': "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history."},
        'meta-llama/Llama-3.1-8B-Instruct' : {'role': 'system', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history. Do not add a header.", 'gender':  "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history."},
        'google/gemma-2-9b-it': {'role': 'user', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history. Do not add a header.", 'gender':  "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history."},
        'allenai/OLMo-2-0325-32B-Instruct' : {'role':'user', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history.", 'gender': "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history."}
    }


    text_template = text_templates[model_name][demographic_type]
    role = text_templates[model_name]['role']

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LanguageModel(model_name, dtype=torch.bfloat16, load_in_4bit=True, device_map='auto')

    system = "You are a helpful clinical assistant."


    messages = [
            {"role": role, "content": "The patient is {0}.".format(target_condition)},
    ]

    target_chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    text = text_template.format(source_condition)
    messages = [
                {"role": role, "content": "{0}\n\n{1}".format(system, text)},
    ]
    source_chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    prompts = [target_chat_text, source_chat_text]

    prompt_id = 0
    corrupt_id = 1
    N_LAYERS=len(llm.model.layers)

    patch_layers = [layer]
    if window>0:
        for k in range(1,window+1):
            front = layer+k
            back = layer-k
            if front>=0 and front<N_LAYERS:
                patch_layers.append(front)
            if back>=0 and back<N_LAYERS:
                patch_layers.append(back)

    print('Final patch layers: ', patch_layers)


    clean_tokens = llm.tokenizer(prompts[0], return_tensors="pt")["input_ids"][0]
    corrupted_tokens = llm.tokenizer(prompts[1], return_tensors="pt")["input_ids"][0]

    target_condition_ids = llm.tokenizer(' ' + target_condition, return_tensors="pt")['input_ids'][0]
    source_condition_ids = llm.tokenizer(' ' + source_condition, return_tensors="pt")['input_ids'][0]

    diff = len(clean_tokens) - len(corrupted_tokens)
    print(diff)

    target_ix, source_ix = -1, -1

    if target_condition == 'prostate cancer':
        target_ix = -2
        print('Changing target index', target_ix)

    if (model_name in ['allenai/OLMo-7B-0724-Instruct-hf', 'allenai/OLMo-2-0325-32B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct']) and source_condition == 'sarcoidosis':
        source_ix = -2
        print('Changing source index', source_ix)

    if model_name == 'allenai/OLMo-7B-0724-Instruct-hf' and source_condition == 'rheumatoid arthritis':
        source_ix = -2
        print('Changing source index', source_ix)

    patch_token_from = torch.argwhere(clean_tokens == target_condition_ids[target_ix])[0][0].tolist()
    patch_token_to = torch.argwhere(corrupted_tokens == source_condition_ids[source_ix])[0][0].tolist()

    #adjust index since prompts will be padded
    if diff>0:
        patch_token_to = patch_token_to + diff
        
    else:
        patch_token_from = patch_token_from - diff
        
    print('from:', patch_token_from, 'to:', patch_token_to)    


    print('Final patch tokens to:', patch_token_to)


    temperature=0.7

    #for interchange accuracy
    max_new_tokens = 80
    outer_N = 25
    inner_N = 20 


    generate_kwargs = dict(do_sample=True, temperature=temperature, top_k=0, top_p=None)

    all_factors = []
    all_texts = []

    factors = [1,2,5]

    if demographic_type == 'race' and window>0:
        factors = [1]

    print('Factors: ', factors)


    for k in range(outer_N):
        print('Batch', k)
        patched_outputs = []
        with torch.no_grad():
            with llm.generate(max_new_tokens=max_new_tokens, **generate_kwargs) as tracer:
                with tracer.invoke(prompts[prompt_id]) as invoker:
                    z_hs = {}
                    for layer_idx in range(N_LAYERS):
                        z = llm.model.layers[layer_idx].mlp.down_proj.output 
                        z_hs[layer_idx] = z[:, patch_token_from, :]

        
                for iter_ix in range(inner_N):
                    for f in factors:
                        with tracer.invoke(prompts[corrupt_id]) as invoker:
                            for layer_idx in patch_layers:
                                z_corrupt = llm.model.layers[layer_idx].mlp.down_proj.output
                                z_corrupt[:,patch_token_to,:] = z_hs[layer_idx]*f
                                llm.model.layers[layer_idx].mlp.down_proj.output = z_corrupt
                            
                            patched_outputs.append(llm.generator.output.save())
                            all_factors.append(f)
        
        for ix in range(len(patched_outputs)):
            text = llm.tokenizer.batch_decode(patched_outputs[ix].value)[0]
            all_texts.append(text)

N = len(all_factors)
df = pd.DataFrame.from_dict({'factor': all_factors, 'text': all_texts, 'layer': [layer]*N, 'window': [window]*N})
df['target'] = target
df['is_success'] = df.apply(is_success, axis=1)

print(df.groupby(['factor'])['is_success'].mean().reset_index().to_markdown())

suffix = model_name.split('/')[-1]
path = os.path.join(output_dir, 'IA_{0}_{1}_l{2}_w{3}_{4}.csv'.format(source_condition, target, layer, window, suffix))

print(path)
df.to_csv(path, sep='\t', index=False)






    
