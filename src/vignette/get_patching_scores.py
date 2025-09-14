import os
import json
import pickle
import torch
import nnsight
import kaleido
import plotly.express as px
from nnsight import LanguageModel, util
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
import plotly.io as pio

def plot_ioi_patching_results(ioi_patching_results,
                              x_labels,
                              y_labels,
                              plot_title="",
                              path = ''):

    fig = go.Figure()
    fig.write_image("random.pdf")
    
    fig = go.Figure()

    fig = px.imshow(
        ioi_patching_results,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Token", "y": "Layer","color":" "},
        x=x_labels,
        y=y_labels,
        title=plot_title,
    )

    if path != '':
        pio.write_image(fig, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch gender/race")
    parser.add_argument('-demographic_type', choices=['gender', 'race'], help='demographic')
    parser.add_argument('-sexed_condition', action='store_true', help='only valid for gender. whether to patch using sexed condition')
    parser.add_argument('-condition', type=str, help='clinical condition')
    parser.add_argument('-source', type=str, help='race to flip from')
    parser.add_argument('-target', type=str, help='race/gender to flip to')
    parser.add_argument('-model_name', choices=['allenai/OLMo-7B-0724-Instruct-hf', 'meta-llama/Llama-3.1-8B-Instruct', 'google/gemma-2-9b-it', 'allenai/OLMo-2-0325-32B-Instruct'], help='model type', default='allenai/OLMo-7B-Instruct')
    parser.add_argument('-output_dir', type=str)



    args = parser.parse_args()
    demographic_type = args.demographic_type
    source_condition = args.condition
    model_name = args.model_name
    target = args.target.strip().lower()
    source = args.source.strip().lower() if demographic_type == 'race' else ''
    sexed_condition = True if args.sexed_condition else False
    output_dir = args.output_dir 

    if sexed_condition:
        assert demographic_type == 'gender'


    scores_path = os.path.join(output_dir, f'{demographic_type}-{source_condition}-{target}-{model_name}-patch_scores.p'.replace('/', '-'))
    plot_path = os.path.join(output_dir, f'{demographic_type}-{source_condition}-{target}-{model_name}-patch_scores_plot.pdf'.replace('/', '-'))

    print('Demographic Type:', demographic_type)
    print('Clinical condition:', source_condition)
    print('Source:', source)
    print('Target:', target)


    text_templates = {
        'allenai/OLMo-7B-0724-Instruct-hf': {'role': 'user', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history. You must start with the following: \"Race:\".", 'gender': "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history. You must start with the following: \"Gender:\"."},
        'meta-llama/Llama-3.1-8B-Instruct' : {'role': 'system', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history. You must start with the following: \"Race:\". Do not add a header.", 'gender':  "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history. You must start with \"Gender:\". Do not add a header."},
        'google/gemma-2-9b-it': {'role': 'user', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history. You must start with the following: \"Race:\". Do not add a header.", 'gender':  "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history. You must start with \"Gender:\". Do not add a header."},
        'allenai/OLMo-2-0325-32B-Instruct' : {'role':'user', 'race': "Please provide a concise example patient presentation of a patient presenting with {0}, including complete demographic details and relevant past medical history. You must start with the following: \"Race:\".", 'gender': "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history. You must start with the following: \"Gender:\"."}
    }


    text_template = text_templates[model_name][demographic_type]
    role = text_templates[model_name]['role']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LanguageModel(model_name, dtype=torch.bfloat16, device_map='auto', load_in_4bit=True,) #load_in_4bit=True,

    system = "You are a helpful clinical assistant."

    if demographic_type == 'gender':
        if target == 'male':
            answers = [(' Male', ' Female')]
        else:
            answers = [(' Female', ' Male')]

        if not sexed_condition:
            print('Patching using gender-explicit prompt')

            if target == 'male':
                target_condition = 'Male'
            else:
                target_condition = 'Female'

            messages = [
                {"role": role, "content": "The patient is {0}.".format(target_condition)},
            ]

        else:
            print('Patching using gender-implicit (clinical condition) prompt')
            if target == 'male':
                target_condition = 'prostate cancer'
            else:
                target_condition = 'preeclampsia'

            text = text_template.format(target_condition)

            messages = [
                        {"role": role, "content": "{0}\n\n{1}".format(system, text)},
            ]
    else:
        source = source[0].upper() + source[1:].lower()
        target = target[0].upper() + target[1:].lower()

        answers = [(' ' + target, ' ' + source)]

        target_condition = target

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

    print(prompts)

    prompt_id = 0
    corrupt_id = 1
    N_LAYERS = len(llm.model.layers)
    softmax = torch.nn.Softmax(dim=-1)

    clean_tokens = llm.tokenizer(prompts[0], return_tensors="pt")["input_ids"][0]
    corrupted_tokens = llm.tokenizer(prompts[1], return_tensors="pt")["input_ids"][0]

    clean_decoded_tokens = [llm.tokenizer.decode(token) for token in clean_tokens]
    print(clean_decoded_tokens)

    corrupted_decoded_tokens = [llm.tokenizer.decode(token) for token in corrupted_tokens]
    print(corrupted_decoded_tokens)

    target_condition_ids = llm.tokenizer(' ' + target_condition, return_tensors="pt")['input_ids'][0]
    source_condition_ids = llm.tokenizer(' ' + source_condition, return_tensors="pt")['input_ids'][0]

    diff = len(clean_tokens) - len(corrupted_tokens)
    print(diff)

    
    target_ix, source_ix = -1, -1

    if target_condition == 'prostate cancer':
        target_ix = -2
        print('Changing target index', target_ix)

    patch_token_from = torch.argwhere(clean_tokens == target_condition_ids[target_ix])[0][0].tolist() #-2 because 'prostate' is what mostly captureness maleness (not cancer, which would been indexed by -1, as is typically how you would index the last subject token)
    offset = 0 

    #adjust index since prompts will be padded
    if diff>0:
        offset = diff
        print('Corrupted prompt is shorter, so will be padded. Need to adjust token index by offset', offset)
        
    else:
        patch_token_from = patch_token_from - diff
        print('Clean prompt is shorter, so will be padded. Need to adjust patch_token_from index by offset', offset)
    

    print('patch_token_from', patch_token_from)
    answer_token_indices = [
                [llm.tokenizer(answers[i][j], add_special_tokens = False)["input_ids"][0] for j in range(2)]
                for i in range(len(answers))
        ]
    print("answer_tokens = " , answer_token_indices)
    print(answers)


    z_hs = {}
    rewrite_scores = []
    step=5

    patched_preds, corrupted_preds = [], []


    if model_name == 'google/gemma-2-9b-it':
        step = 2
    elif model_name == 'allenai/OLMo-2-0325-32B-Instruct':
        step = 1

    for start in range(0, N_LAYERS, step):
        end = min(start + step, N_LAYERS)
        print(start, end)
    

        with torch.no_grad():
            with llm.generate(max_new_tokens=6) as tracer:
                with tracer.invoke(prompts[prompt_id]) as invoker:
                    z_hs = {}
                    for layer_idx in range(N_LAYERS):
                        z = llm.model.layers[layer_idx].mlp.down_proj.output
                        z_hs[layer_idx] = z[:, patch_token_from, :]

                                            
                with tracer.invoke(prompts[corrupt_id]) as invoker:
                    if demographic_type == 'race' and model_name == 'allenai/OLMo-7B-0724-Instruct-hf':
                        corrupted_logits = llm.lm_head.next().next().next().output
                    else:
                        corrupted_logits = llm.lm_head.next().next().output
                        
                    corrupted_pred = corrupted_logits[0].argmax(dim=-1).save()
                    corrupted_prob = softmax(corrupted_logits[0][0])[answer_token_indices[prompt_id][0]]
                    corrupted_preds.append(corrupted_pred)
            
                for layer_idx in range(start, end):
                    for token_idx in range(len(corrupted_tokens)):
                        with tracer.invoke(prompts[corrupt_id]) as invoker:
                            z_corrupt = llm.model.layers[layer_idx].mlp.down_proj.output
                            z_corrupt[:,token_idx+offset,:] = z_hs[layer_idx]
                            llm.model.layers[layer_idx].mlp.down_proj.output = z_corrupt

                            if demographic_type == 'race' and model_name == 'allenai/OLMo-7B-0724-Instruct-hf':
                                patched_logits = llm.lm_head.next().next().next().output
                            else:
                                patched_logits = llm.lm_head.next().next().output
                    
                            patched_logit_diff = (
                                                        patched_logits[0, -1, answer_token_indices[prompt_id][0]]
                                                        - patched_logits[0, -1, answer_token_indices[prompt_id][1]]
                                                    )
                                        
                            patched_pred = patched_logits[0].argmax(dim=-1).save()

                    
                            patched_prob = softmax(patched_logits[0][0])[answer_token_indices[prompt_id][0]]
                                    
                            rewrite_score = (patched_prob - corrupted_prob)/(1-corrupted_prob)
                            rewrite_scores.append(rewrite_score.save())
                            patched_result = (patched_prob - corrupted_prob)

                            patched_preds.append(patched_pred)
        

    rewrite_scores = [score.cpu().float().item() for score in rewrite_scores]
    rewrite_scores = np.array(rewrite_scores)
    rewrite_scores = np.reshape(rewrite_scores, (N_LAYERS, -1))

    corrupted_decoded_tokens = [llm.tokenizer.decode(token) for token in corrupted_tokens]
    token_labels = [f"{token}_{index}" for index, token in enumerate(corrupted_decoded_tokens)]
    layer_labels = [index for index in range(len(rewrite_scores))]

    patched_preds =  [p.cpu().float().item() for p in patched_preds]

    results = {}
    results['token_labels'] = token_labels
    results['layer_labels'] = layer_labels
    results['rewrite_scores'] = rewrite_scores
    results['patched_preds'] = patched_preds

    with open(scores_path, 'wb') as f:
        pickle.dump(results, f)

    #Note: skipping plotting patching results in layer 0
    plot_ioi_patching_results(results['rewrite_scores'][1:,:], results['token_labels'], results['layer_labels'][1:], '', plot_path)









        











