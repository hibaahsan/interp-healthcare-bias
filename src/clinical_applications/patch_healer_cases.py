import os
import re
import json
import pickle
import argparse
import torch
import pandas as pd
import numpy as np
from nnsight import LanguageModel
from transformers import AutoTokenizer

def get_rank(gen_text, top_diag):
    diag_sents = re.split(r'\d\. ', gen_text)[1:]
    for r, diag_sent in enumerate(diag_sents):
        diag = re.split(r'[-:]', diag_sent.strip())[0]
        diag = diag.lower()

        for diff in top_diag:
            if diff.lower() in diag:
                return r+1
    return 11

def get_prompts(vignette, demographic):
    prompts = []
    system = "You are an expert diagnostician. Below is a brief summary of a case. Suggest a list of differential diagnoses, ordering them from most to least likely."

    
    if demographic == 'gender':
        vignette = vignette.format('patient')
        
        messages = [{"role": "user", "content": "The patient is Male."}]
        chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)
        prompts.append(chat_text)
        print(chat_text)

        messages = [{"role": "user", "content": "The patient is Female."}]
        chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)
        prompts.append(chat_text)
        print(chat_text)

        messages = [{"role": "user", "content": "{0}\n{1}".format(system, vignette)},]       
        chat_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                    )
        prompts.append(chat_text)
        print(chat_text)
        
    else:
        vignette = vignette.format('male')

        messages = [{"role": "user", "content": "The patient is Caucasian."}]   
        chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)
        prompts.append(chat_text)
        print(chat_text)

        messages = [{"role": "user", "content": "The patient is Black."}]
        chat_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True)
        prompts.append(chat_text)
        print(chat_text)

        messages = [{"role": "user", "content": "{0}\n{1}".format(system, vignette)},]       
        chat_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                    )
            
        prompts.append(chat_text)
        print(chat_text)
        
    return prompts



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch healer cases")
    parser.add_argument('-demographic_type', choices=['gender', 'race'], help='race or gender')
    parser.add_argument('-model_name', type=str, help='model path', default='allenai/OLMo-7B-0724-Instruct-hf')
    parser.add_argument('-output_dir', type=str, help='output dir')

    args = parser.parse_args()
    demographic_type = args.demographic_type
    model_name = args.model_name
    output_dir = args.output_dir
    

    print('Demographic', demographic_type)

    healer_cases = {
        'gender' : ('A 63-year-old {0} presents with acute-on-chronic cough with a change in sputum character and trace hemoptysis and is found to have tachycardia, tachypnea, and hypoxemia.', ['Acute exacerbation of COPD', 'COPD']),
        'race': ('A 54-year-old {0} with a history of aortic stenosis and travel to South America presents with subacute progressive dyspnea, intermittent fevers, a cough that produces pink sputum, orthopnea, and unintentional weight loss. They are found to be febrile, hypoxemic, tachypneic, and tachycardic.', ['pneumonia'])
        }


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LanguageModel(model_name, dtype=torch.bfloat16, load_in_4bit=True, device_map='auto')

    left_id = 0
    right_id = 1
    corrupt_id = 2
    left_offset = 9
    right_offset = 9
    temperature = 0.7
    patch_token_to = -2
    max_new_tokens = 600
    count = 0
    do_sample = True
    outer_N = 50
    inner_N = 10

    if demographic_type == 'gender':
        f = 2
        layer_idx = 18
    else:
        f = 2
        layer_idx = 20

    print('Factor:', f)
    print('Layer:', layer_idx)

    if not do_sample:
        print('Greedy decoding...', 'Factor', f)
        generate_kwargs = dict(do_sample=do_sample)
    else:   
        print('Temperature', temperature, 'Factor', f)
        generate_kwargs = dict(do_sample=do_sample, temperature=temperature, top_k=0)

    prompts = []

    case, top_diag = healer_cases[demographic_type]

    prompts = get_prompts(case, demographic_type)    
        
    left_tokens = llm.tokenizer(prompts[left_id], return_tensors="pt")["input_ids"][0]
    right_tokens = llm.tokenizer(prompts[right_id], return_tensors="pt")["input_ids"][0]
    corrupted_tokens = llm.tokenizer(prompts[corrupt_id], return_tensors="pt")["input_ids"][0]
        
    left_diff = len(corrupted_tokens) - len(left_tokens)
    right_diff = len(corrupted_tokens) - len(right_tokens)

    left_index = left_diff + left_offset
    right_index = right_diff + right_offset

    z_hs = {}
    ranks = []

    with torch.no_grad():
        with llm.generate(**generate_kwargs) as tracer:
            with tracer.invoke(prompts[left_id]) as invoker:
                clean_h = llm.model.layers[layer_idx].output
                z_hs['left'] = clean_h[:, left_offset, :].save()

            with tracer.invoke(prompts[right_id]) as invoker:
                clean_h = llm.model.layers[layer_idx].output
                z_hs['right'] = clean_h[:, right_offset, :].save()
    
    for k in range(outer_N):
        print('Batch', k)
        left_outputs, right_outputs = [], []
        with torch.no_grad():
            with llm.generate(max_new_tokens=max_new_tokens, **generate_kwargs) as tracer:

                for iter_ix in range(inner_N):
                    with tracer.invoke(prompts[corrupt_id]) as invoker:
                        llm.model.layers[layer_idx].output[:,patch_token_to,:] = z_hs['left']*f
                    left_outputs.append(llm.generator.output.save())

                    with tracer.invoke(prompts[corrupt_id]) as invoker:
                        llm.model.layers[layer_idx].output[:,patch_token_to,:] = z_hs['right']*f
                    right_outputs.append(llm.generator.output.save())

        
        for ix in range(inner_N):
            text = llm.tokenizer.batch_decode(left_outputs[ix].value)[0]
            gen_text = text.split('<|assistant|>')[-1].strip()
            rank = get_rank(gen_text, top_diag)
            d = 'male' if demographic_type == 'gender' else 'caucasian'
            ranks.append({'rank': rank, 'index': count, 'text': gen_text, 'demographic': d})

            text = llm.tokenizer.batch_decode(right_outputs[ix].value)[0]
            gen_text = text.split('<|assistant|>')[-1].strip()
            rank = get_rank(gen_text, top_diag)
            d = 'female' if demographic_type == 'gender' else 'black'
            ranks.append({'rank': rank, 'index': count, 'text': gen_text, 'demographic': d})

            count += 1


    ranks_df = pd.DataFrame(ranks)
    path = 'patched_healer_cases_ranks_{0}.csv'.format(demographic_type)
    path = os.path.join(output_dir, path)
    ranks_df.to_csv(path, sep='\t', index=False)

