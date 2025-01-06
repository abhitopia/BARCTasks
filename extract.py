#%%
import json
from pathlib import Path
import jsonlines
from tqdm import tqdm

def convert_to_arc_format(pair_list):
    if not isinstance(pair_list, list):
        raise ValueError('pair_list is not a list.')
    train_list = []
    test_list = []
    if len(pair_list) < 10:
        number_of_tests = 1
    elif len(pair_list) < 20:
        number_of_tests = 2
    else:
        number_of_tests = 3
    for pair_index, pair_item in enumerate(pair_list):
        # print(f'Item {pair_index}: {pair_item} {len(pair_item)}')
        if len(pair_item) != 2:
            raise ValueError(f'Item {pair_index} does not have 2 elements.')
        input_image = pair_item[0]
        output_image = pair_item[1]
        dict = {'input': input_image, 'output': output_image}
        if pair_index >= len(pair_list) - number_of_tests:
            test_list.append(dict)
        else:
            train_list.append(dict)
    arc_dict = {'train': train_list, 'test': test_list}
    return arc_dict



def extract_from_jsonl(path, max_file_size=49 * 1024 * 1024):
    progress_bar = tqdm(desc='Extracting', unit=' tasks', total=100000)

    output_folder = Path(path).parent / Path(path).stem
    output_folder.mkdir(exist_ok=True)

    current_file_idx = 0
    current_file_size = 0
    current_file_path = output_folder / f'{current_file_idx}.jsonl'
    current_file = current_file_path.open('w')

    with jsonlines.open(path) as reader:
        for idx, obj in enumerate(reader):
            progress_bar.update(1)
            examples = obj['examples']
            arc_dict = convert_to_arc_format(examples)
            json_str = json.dumps(arc_dict) + '\n'
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > max_file_size:
                print(f"Skipping large json_str at index {idx}")
                continue

            if current_file_size + len(json_bytes) > max_file_size:
                current_file.close()
                current_file_idx += 1
                current_file_path = output_folder / f'{current_file_idx}.jsonl'
                current_file = current_file_path.open('w')
                current_file_size = 0

            current_file.write(json_str)
            current_file_size += len(json_bytes)

    current_file.close()

# %%
# path = "data/100k_gpt4o-mini_generated_problems.jsonl"
# extract_from_jsonl(path)

# path = "data/100k-gpt4-description-gpt4omini-code_generated_problems.jsonl"
# extract_from_jsonl(path)

# path = 'data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems_data_100k.jsonl'
# extract_from_jsonl(path)

path = 'data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems_data_suggestfunction_100k.jsonl'
extract_from_jsonl(path)
# %%
