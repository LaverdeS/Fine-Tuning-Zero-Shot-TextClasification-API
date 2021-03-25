import json
import pandas as pd
import torch
import requests

from enum import Enum
from pydantic import BaseModel
from torch.utils.data import DataLoader


def backup_responses(post_out: dict):
    with open('./log.txt', "a") as f:
        json.dump(post_out, f)
        f.write('\n')
    return


def booksummaries(trimmer_ts=30):  # trimmer <30 and work only with 100
    # trimmer_ts is the minimum lenght for the Plot feature
    with open('../booksummaries/booksummaries.txt', 'r', encoding='utf-8') as dataset:
        df = pd.read_csv(dataset, sep='\t', header=None)
        df.columns = ['WikiID', 'FreebaseID', 'Title', 'Author', 'Year', 'Genre', 'Plot']
        df.drop(columns=['WikiID', 'FreebaseID'], inplace=True)

    for i, sample in enumerate(df.Genre):
        if str(sample) == 'nan':
            continue
        genre_i = str(sample).replace('"', '').replace(',', '').replace('}', '').replace("'", '').split(' ')
        genre_i_ = [part for part in genre_i if part.isalpha()]
        # print(genre_i_)
        genre_fixi = []
        ixx = 0
        for ix, word in enumerate(genre_i_):
            if ix == 0 and word[0].islower():
                word = word[0].upper() + word[0:]
            if word[0].isupper():
                genre_fixi.append(word)
            if word[0].islower():
                ixx += 1
                genre_fixi[ix - ixx] = genre_fixi[ix - ixx] + ' ' + word
        # print(genre_fixi)
        df.Genre[i] = genre_fixi

    for i, sample in enumerate(df.Year):
        if len(str(sample)) > 4:
            df.Year[i] = sample[0:4]

    df.dropna(inplace=True)
    print(f'lenght before trimmering shorties: {len(df)}')
    df = df[df['Plot'].str.len() >= trimmer_ts]
    print(f'lenght after trimmering shorties: {len(df)}')
    return df.iloc[5:8, :]


def booksummaries_range(from_: int, to_: int, trimmer_ts=30):  # trimmer <30 and work only with 100
    # trimmer_ts is the minimum lenght for the Plot feature
    with open('../booksummaries/booksummaries.txt', 'r', encoding='utf-8') as dataset:
        df = pd.read_csv(dataset, sep='\t', header=None)
        df.columns = ['WikiID', 'FreebaseID', 'Title', 'Author', 'Year', 'Genre', 'Plot']
        df.drop(columns=['WikiID', 'FreebaseID'], inplace=True)

    for i, sample in enumerate(df.Genre):
        if str(sample) == 'nan':
            continue
        genre_i = str(sample).replace('"', '').replace(',', '').replace('}', '').replace("'", '').split(' ')
        genre_i_ = [part for part in genre_i if part.isalpha()]
        # print(genre_i_)
        genre_fixi = []
        ixx = 0
        for ix, word in enumerate(genre_i_):
            if ix == 0 and word[0].islower():
                word = word[0].upper() + word[0:]
            if word[0].isupper():
                genre_fixi.append(word)
            if word[0].islower():
                ixx += 1
                genre_fixi[ix - ixx] = genre_fixi[ix - ixx] + ' ' + word
        # print(genre_fixi)
        df.Genre[i] = genre_fixi

    for i, sample in enumerate(df.Year):
        if len(str(sample)) > 4:
            df.Year[i] = sample[0:4]

    df.dropna(inplace=True)
    print(f'lenght before trimmering shorties: {len(df)}')
    df = df[df['Plot'].str.len() >= trimmer_ts]
    print(f'lenght after trimmering shorties: {len(df)}')
    return df.iloc[from_: to_, :]


def create_sentences(target: str, sentence_lenght: int, verbose=False):
    '''sentence_length is actually a minimum; there is no maximum. The end is defined by the dot here for consistency.
  it tries to build a sentence of the minimum lenght and checks if ends in a dot. If not, then extend the lenght +1 and check again. Do that until finding a dot.
  if sentence_lenght = 1 --> split until finding dot for the minimum sentence lenght possible (one single word followed by a dot would work)
  if sentence_lenght is too large --> might get a list of one item: one very long sentence.
  I need to define a sentence_lenght and then reduce it about 15 words in order to avoid getting sentences which are too long.
  the sentence_lenght would then be = num_words_in_page/2 - 15; num_words_in_page ~ '''

    target_list = target.replace('\n', '').replace('"', '').replace("'", "").split(' ')
    chunk_size = sentence_lenght - 1
    chunks = []
    start = 0
    end = start + chunk_size + 1
    while end <= len(target_list):
        chunk_i = target_list[start:end]
        while '.' not in chunk_i[-1] and end != len(target_list):
            chunk_i.append(target_list[end])
            if len(chunk_i) > 125:
                break
            end = end + 1
        start = end
        end = end + chunk_size + 1
        chunks.append(' '.join(chunk_i))
    if verbose:
        for ix, paragr in enumerate(chunks):
            print(f'paragraph {ix + 1}:\n{paragr}\n')
    return chunks


def classify(paragraphs, labels, classifier, top_n, threshold, multilabel=False, single=False):
    print("classifying")
    original_lenght = len(paragraphs)
    # try catch RuntimeError here
    if not multilabel:
        if single:
            print("single inference:")
            # paragraphs will be a single string, not list
            output_dict = classifier(paragraphs, labels, multi_class=multilabel)
            return output_dict
        try:
            output_dict = classifier(paragraphs, labels, multi_class=multilabel)
            df = pd.DataFrame(output_dict)
            df.drop(columns=['sequence'], inplace=True)
            df_threshold = df.copy(deep=True)
            df_threshold.scores = df_threshold.scores.apply(
                lambda x: str(x).replace('[', '').replace(']', ''))  # join btter
            df_threshold = df_threshold[df_threshold['scores'].astype('float16') >= float(threshold) / 100.0]

        except RuntimeError:
            print("\nCUDA device run out of memory. Building lighter individual classifiers. This may take a while...")
            out_df = pd.DataFrame()

            for ix, paragraph in enumerate(paragraphs):
                output_dict = classifier(paragraph, labels, multi_class=multilabel)
                df = pd.DataFrame(output_dict)
                df.drop(columns=['sequence'], inplace=True)
                out_df = pd.concat([out_df, df], axis=0, ignore_index=True)  # , keys=[]) # test_toy this first!
                df_threshold = out_df.copy(deep=True)
                df_threshold = df_threshold[df_threshold['scores'].astype('float16') >= float(threshold) / 100.0]
                df = out_df

    if multilabel:
        # print('paragraph: ', sequence, '\n', df[0:top_n], '\n')
        # here implement for multilabel return.
        pass

    output_lenght = len(df)
    print(f'number of paragraphs removed because of threshold: {original_lenght - output_lenght}')
    print(f'number of paragraphs in the output DataFrame: {output_lenght}')
    return df, df_threshold


def print_dict(x):
    print("")
    for k, v in x.items():
        print(f"{k}:{v}")
    print("")


class InferenceConfig(BaseModel):
    single_inference: bool = False
    description: str = 'zero-shot using BART-MNLI'
    paragraph_size: int = 40
    threshold: float = 65.0
    text: str = ""


class RequestTarget(BaseModel):
    data: str = 'demo'
    description: str = 'zero-shot using BART-MNLI'
    paragraph_size: int = 40
    threshold: float = 80.0


class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class LabelsFilter(str, Enum):
    pos = "positives"
    neg = "negatives"
    all_ = "all"


def get_training_text(data_start=20, data_end=25):
    training_df = booksummaries_range(from_=data_start, to_=data_end)
    training_text = training_df['Plot'].tolist()

    # now let's create shorter sentences with the paragraphs.
    training_text = [create_sentences(paraf, sentence_lenght=40) for paraf in training_text]

    # flatten
    training_text = [chunk for paraf in training_text for chunk in paraf]
    print(f"training_text lenght: {len(training_text)}")
    return training_text


def generate_labeled_data():
    # I will automatically generate the labels with the same classifier but different text and text_size with labels pushed to 1.0
    # or 0.0 with threshold 65%

    # loop to make post-requests to api
    training_text = get_training_text()

    responses = []
    for ix, sample_text in enumerate(training_text):
        payload = {
            "single_inference": "true",
            "description": "zero-shot using BART-MNLI",
            "paragraph_size": 40,
            "threshold": 65,
            "text": sample_text
        }
        print(f"classifying --> {payload['text'][0:50] + ' ...'}")
        r = requests.post("http://127.0.0.1:8000/api/zero-shot/1", data=json.dumps(payload))
        print(r.url)
        print(f"{ix + 1}: status_code : {r.status_code}\n")
        responses.append(r.json())

    # training_scores = training_df['scores'].tolist()

    training_scores = [r['data']['output'] for r in responses]
    training_pairs = [(paraf, float(score)) for paraf, score in zip(training_text, training_scores)]

    # Now I push them to 1.0 or 0.0 depending on score and threshold

    # or just rounding the label (score)

    # bias-pusher strategy
    rounded_training_pairs = []
    for pair in training_pairs:
        new_pair = (pair[0], float(round(pair[1])))
        rounded_training_pairs.append(new_pair)

    return rounded_training_pairs


def compare(infer_output, output_2, rounded_training_pairs):
    text_dict = {}
    training_labels_dict = {}

    training_labels = [pair[1] for pair in rounded_training_pairs]
    training_plots = [pair[0] for pair in rounded_training_pairs]

    infer_output_dict = {ix: {'scores': i, 'Plot': p} for ix, (i, p) in enumerate(zip(infer_output, training_plots))}

    # for ix, p in enumerate(training_plots):
    #     infer_output_dict

    # model-hub
    model_hub_dict = {}
    for ix, (k, v) in enumerate(infer_output_dict.items()):
        model_hub_dict.update({k: float(v['scores'])})
        text_dict.update({k: v['Plot']})
        training_labels_dict.update({k: training_labels[ix]})  # round..

    # fine-tuned
    fine_tuned_dict = {}
    for k, v in output_2.items():
        fine_tuned_dict.update({k: float(v['scores'][0])})

    # comparison
    comparison_dict = {}

    for k, mh_v in model_hub_dict.items():
        diff = fine_tuned_dict[str(k)] - mh_v
        comparison_dict.update({k: {'text': text_dict[k],
                                    'train_label': training_labels_dict[k],
                                    'change_abs': abs(diff),
                                    'change': str(round(diff * 100.0, 3)) + '%' if diff < 0.0 else '+' + str(
                                        round(diff * 100.0, 3)) + '%'}})

    comparison_dict = dict(sorted(comparison_dict.items(), key=lambda x: x[1]['change_abs'], reverse=True))
    df = pd.DataFrame.from_dict(comparison_dict, orient='index')
    return comparison_dict, df