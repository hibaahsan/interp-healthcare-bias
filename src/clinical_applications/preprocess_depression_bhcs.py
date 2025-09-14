import os
import pickle
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def contains_sexed_condition(text):
    terms = ['ovar', 'pregna', 'breast', 'menop', 'labor', 'pcos', 'cerv', 'endomet', 'menst', 'estrogen', 'transgender', 'hormon', 'contrace', 'vagin', 'uter', 'birth']

    for term in terms:
        if term in text.lower() or 'IUD' in text: 
            return True
    
    return False

def get_token_count(text):
    return len(tokenizer(text)['input_ids'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get BHCs of female patients with anxiety")
    parser.add_argument('-summaries_path', type=str, help='path to input BHCs')
    parser.add_argument('-patients_path', type=str, help='path to patients.csv.gz')
    parser.add_argument('-output_dir', type=str, help='path to output preprocessed BHCs val and test')
    parser.add_argument('-model_type', type=str, help='model type', default='allenai/OLMo-7B-0724-Instruct-hf')

    args = parser.parse_args()
    summaries_path = args.summaries_path
    patients_path = args.patients_path
    output_dir = args.output_dir

    np.random.seed(42)

    N = 1000
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    
    note_di_df = pd.read_csv(summaries_path)
    patients_df = pd.read_csv(patients_path)

    patients_df = patients_df[patients_df['gender'] == 'F']
    patients_df = patients_df[['subject_id', 'gender']].drop_duplicates()

    note_di_df = note_di_df.merge(patients_df, on=['subject_id'])
    print('Number of rows after filtering for female patients:', len(note_di_df))

    note_di_df['is_sexed'] = note_di_df['brief_hospital_course'].apply(contains_sexed_condition)
    note_di_df = note_di_df[note_di_df['is_sexed']==False].copy()
    print('Number of rows after filtering BHCs with sexed conditions:', len(note_di_df))

    note_di_df = note_di_df[(note_di_df['brief_hospital_course'].str.lower().str.contains('anxiety'))&(~note_di_df['brief_hospital_course'].str.lower().str.contains('depression'))]
    print('Number of rows after filtering for BHCs with term anxiety but not depression:', len(note_di_df))

    note_di_df['tok_count'] = note_di_df['brief_hospital_course'].apply(get_token_count)
    note_di_df = note_di_df[note_di_df['tok_count']<2000]

    print('Number of rows after filtering BHCs with >2000 tokens:', len(note_di_df))

    note_di_df = note_di_df.sample(frac=1.0)

    all_bhcs = list(note_di_df['brief_hospital_course'])
    test_bhcs = all_bhcs[:N]
    val_bhcs = all_bhcs[N:N+100]

    print('# val BHCs:', len(val_bhcs))
    print('# test BHCs:', len(test_bhcs))

    bhcs_dict = {'val': val_bhcs, 'test': test_bhcs}

    output_path = os.path.join(output_dir, 'bhcs_dataset.p')
    print('Output path:', output_path)

    with open(output_path, 'wb') as f:
        pickle.dump(bhcs_dict, f)