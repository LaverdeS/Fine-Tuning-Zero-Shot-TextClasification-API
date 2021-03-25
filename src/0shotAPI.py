import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = '2'

import uvicorn
import json
import pandas as pd
import torch
from transformers import pipeline
from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from transformers import BartForSequenceClassification, BartTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm


def backup_responses(post_out: dict):
    with open('./log.txt', "a") as f:
        json.dump(post_out, f)
        f.write('\n')
    return


def booksummaries(trimmer_ts=30):  # trimmer <30 and work only with 100
    # trimmer_ts is the minimum lenght for the Plot feature
    with open('booksummaries/booksummaries.txt', 'r', encoding='utf-8') as dataset:
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
            df.Year[i] = sample[5:10]

    df.dropna(inplace=True)
    print(f'lenght before trimmering shorties: {len(df)}')
    df = df[df['Plot'].str.len() >= trimmer_ts]
    print(f'lenght after trimmering shorties: {len(df)}')
    return df.iloc[20:25, :]


def create_sentences(target: str, sentence_lenght: int, verbose=False):
    '''sentence_length is actually a minimum; there is no maximum. The end is defined by the dot here for consistency.
  it tries to build a sentence of the minimum lenght and checks if ends in a dot. If not, then extend the lenght +1 and check again. Do that until finding a dot.
  if sentence_lenght = 1 --> split until finding dot for the minimum sentence lenght possible (one single word followed by a dot would work)
  if sentence_lenght is too large --> might get a list of one item: one very long sentence.
  I need to define a sentence_lenght and then reduce it about 15 words in order to avoid getting sentences which are too long.
  the sentence_lenght would then be = num_words_in_page/2 - 15; num_words_in_page ~ '''

    target_list = target.replace('\n', '').replace('"', '').split(' ')
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
            print(f"result of classify: {len(df)}\n{df.iloc[0:5, :]}")
            df.drop(columns=['sequence'], inplace=True)
            df_threshold = df.copy(deep=True)
            df_threshold.scores = df_threshold.scores.apply(
                lambda x: str(x).replace('[', '').replace(']', ''))  # join btter
            df_threshold = df_threshold[df_threshold['scores'].astype('float16') >= float(threshold) / 100.0]

        except RuntimeError:
            print("\nCUDA device run out of memory. Building lighter individual classifiers. This may take a while...")
            out_df = pd.DataFrame()

            for ix, paragraph in tqdm(enumerate(paragraphs)):
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
    threshold: float = 50.0
    text: str = ""


class TrainingData(BaseModel):
    pairs: list = [("text_1", 1.0), ("text_2", 0.0)]


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


app = FastAPI(debug=True)
db = []
feedback_carrier = []


@app.get("/")
async def index():
    print("\n@ INDEX.")
    return {'key': 'value'}


@app.get("/api/")
async def api_root():
    print("\n@ ROOT_API.")
    return {
        'message': 'Hello there! This is an API that is able to classify any text in any language into a set of arbitratry labels in a zero-shot fashion. Additionally, you can use the model for inference and correct the result with a feedback request and re-train the model.'}


@app.get("/api/zero-shot/")
async def get_zero_shot():
    print("\n@ get_zero_shot_resource.")
    print(f"lenght of current db: {len(db)}")
    db_ = {i: db[i] for i in range(len(db))}
    print(f"db_:{db_}")

    if len(db) > 0:
        out_dt = {k: dict() for k in range(len(db))}
        for db_i, value_dt in zip(db, out_dt.values()):
            update_dt = {k: v for k, v in db_i.items()}
            update_dt['model'] = update_dt['model'].task

            if len(update_dt['data']) > 0:  # parse df inside data
                update_dt['data'] = update_dt['data'].to_json(orient="records")

            print(f"model.task: {update_dt['model']}")
            value_dt.update(update_dt)

    dump_this = json.dumps(db_, indent=4) if len(db) == 0 else json.dumps(out_dt, sort_keys=True, indent=4,
                                                                          ensure_ascii=False)
    print(f"\ndump_this: {dump_this}")
    # dump_this = json.loads(dump_this)
    values = []
    if len(db) > 0:
        dt = {k: [] for k in db[0]}
        for ky in db[0].keys():
            for i in db:
                if ky != "model":
                    dt[ky].append(i[ky])
                elif ky == "model":
                    dt[ky].append(i[ky].task)
        print(f"\ndt: {dt}")
        df_dt = pd.DataFrame.from_dict(dt, orient="index")
        df_dt = df_dt.transpose()
        print(f"\ndataframe from db: \n{df_dt}\n")

        # result = # id of classifier
        result = df_dt.to_json(orient="records")
        result = json.loads(result)
        print(f"\nresult (index for df): \n{result}\n")

    return {'data':
                {'models': 'list of available models',
                 'db': result if len(db) > 0 else dump_this},
            'message': "HuggingFace model-hub | current models"
            }


@app.post("/api/zero-shot/")
async def create_classifier(category: str):  # , pretrained_model: str):
    print("\n@ POST create_classifier_resourse.")
    print("intializing classifier...")
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')  # , device=0)

    global db
    new_id = 1 if len(db) == 0 else db[-1]['cl_id'] + 1
    print(f"new_id created: {new_id}")
    db.append({'cl_id': new_id, 'model': classifier, 'category': category, 'data': []})
    return {'data': None,
            'message': "Model apended to the database. Try the new resource here --> http://127.0.0.1:8000/api/zero-shot/{}".format(
                new_id)}


@app.put("/api/zero-shot/")
async def upload_pretrained_model_to_classifier(category: str):  # , pretrained_model: str):
    print("\n@ PUT upload_pretrained_2_classifier resourse.")
    print("loading local pretrained model...")

    pretrained_model = BartForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="C:/Users/lavml/Documents/SoSe20/nlp/BERT/restapi/results/fine-tuning/",
        local_files_only=True)
    tokenizer = BartTokenizer.from_pretrained('results/fine-tuning/', vocab_file="results/fine-tuning/vocab.json",
                                              merges_file="results/fine-tuning/merges.txt")

    classifier = pipeline('zero-shot-classification', model=pretrained_model, tokenizer=tokenizer)  # , device=0)

    # add it to the db

    new_id = 1 if len(db) == 0 else db[-1]['cl_id'] + 1
    db.append(
        {'cl_id': new_id, 'model': classifier, 'category': category, 'data': [], 'description': 'fine-tuned model'})

    return {'data': None,
            'message': f"Fine-tuned Model apended to the database under the id: {new_id}. Try the new resource here "
                       f"--> http://127.0.0.1:8000/api/zero-shot/{new_id}"}


@app.post("/api/zero-shot/{id_}")  # doesn't allow me to have a body for a get request.
async def run_inference(query: InferenceConfig, id_: int):
    print(f"\n@ inference_resource; id: {id_}")
    request = query.dict()
    print_dict(request)
    global feedback_carrier
    # print(f"this is feedback_carrier @ start of run_inference: {feedback_carrier}\n")
    classifier_ = db[-1]
    classifier = classifier_['model']
    category = classifier_['category']

    print(f"category (last): {category}")

    for classifier_ in db:
        if str(classifier_['cl_id']) == str(id_):
            classifier = classifier_['model']
            category = classifier_['category']
            print(f"found! id: {id_}, category: {category}, classifier: {classifier}")
            break
        print(f"classifier id not found...({id_})")

    if request['single_inference']:
        print("inference check: true")
        infer_out = classify(request['text'], category, classifier, 5, request['threshold'], multilabel=False,
                             single=True)
        # print(infer_out)
        score_ = infer_out['scores']
        print(f"inference score: {score_, type(score_)}")

        decision = 1.0 if score_[0] >= request['threshold'] else 0.0

        if len(feedback_carrier) > 0:
            for load_ in feedback_carrier:
                if '~' in str(load_['id']):
                    print("classifier id has been found for single inference paragraphs objects...")
                    paragraph_key = load_['paragraphs']

                    new_sample = {'text': request['text'],
                                  'decision': decision,
                                  'confidence': score_
                                  }

                    paragraph_key.append(new_sample)
                    print(f"current paragraphs: {paragraph_key}")

                else:
                    feedback_carrier.append({'id': str(id_) + '~',
                                             'threshold': request['threshold'],
                                             'paragraphs': [{'text': request['text'],
                                                             'decision': decision,
                                                             'confidence': score_}],
                                             'positives': [[], []],
                                             'negatives': [[], []],
                                             'feedback': [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), request['threshold']]
                                             })
        else:
            feedback_carrier.append({'id': str(id_) + '~',
                                     'threshold': request['threshold'],
                                     'paragraphs': [{'text': request['text'],
                                                     'decision': decision,
                                                     'confidence': score_}],
                                     'positives': [[], []],
                                     'negatives': [[], []],
                                     'feedback': [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), request['threshold']]
                                     })


        # print(f"\nNew carrier:  {feedback_carrier}")

        return {'data':
                    {'classifier': str(classifier),
                     'category': category,
                     'output': str(score_[0]),
                     'message': f"Sucess! data stored to id_feed: {str(id_) + '~'}. Use this for adding training data."}
                }

    else:
        df = booksummaries()
        print(f'\nexploding dataFrame by paragraphs...')
        print(f'number of samples: {len(df.Plot)}')

        for ix, plot in enumerate(df.Plot):
            sentences = create_sentences(plot, sentence_lenght=int(request['paragraph_size']))
            if len(sentences) > 0:
                df.iloc[ix].Plot = sentences

        df = df.explode('Plot', ignore_index=True)
        paragraphs = df.Plot.tolist()
        print(f'number of paragraphs: {len(paragraphs)}\n')

    df_complete, df_out = classify(paragraphs, category, classifier, 5, request['threshold'], multilabel=False,
                                   single=False)
    positive_indexes = list(df_out.index)
    print("positives: ", positive_indexes)

    # POSITIVES & CONCAT
    print("\nbefore concatenation (pos/total): ", len(df_out), "/", len(df))
    print("\n***before concatenation: \ndf_threshold:\n", df_out, "\n")
    df_complete = pd.concat([df, df_complete], axis=1)
    df_complete.drop(['Year', 'Genre'], axis=1, inplace=True)
    df = pd.concat([df, df_out], axis=1)  # positive df comes with label and score extras
    print("concat df: \n", df.head())
    df.fillna(value=False, inplace=True)
    df.drop(['Author', 'Year', 'Genre'], axis=1, inplace=True)
    df_pos = df[df['labels'] != False]
    df_pos.to_csv("temp_df_out_streamlit_positives_fastapi.csv", index=False)
    indexes_original = list(df.index)
    print("\npos df: \n", df_pos.head())
    print("\noriginal indexes: ", indexes_original)
    print("positive indexes: ", positive_indexes)

    # NEGATIVES
    negative_indexes = [ix for ix in indexes_original if ix not in positive_indexes]
    print("negative indexes: ", negative_indexes)
    df_neg = df[df['labels'] == False]
    print("\ndf_neg:\n", df_neg, len(df_neg))
    df_neg.to_csv("temp_df_out_streamlit_negatives_fastapi.csv", index=False)

    # BALANCE DATASET FOR FEEDBACK
    df_neg = df_neg.sample(n=len(df_pos), replace=True)
    feedback_df = pd.concat([df_pos, df_neg], axis=0).sample(frac=1)  # shuffle df
    feedback_df.fillna(value="no value", inplace=True)

    labels_list_list = df_pos['labels'].tolist()
    label_ = next(iter(labels_list_list))
    print(f"label found: {label_}\nfeedback_df:\n{feedback_df}\n")

    feedback_df.to_csv("temp_df_out_streamlit_feedback_fastapi.csv", index=False)

    result = df_complete.to_json(orient="index")  # df to json string
    parsed = json.loads(result)  # json to python dict

    feedback_carrier.append({'id': id_,
                             'positives': [df_pos['Plot'].tolist(), df_pos['scores'].tolist()],
                             'negatives': [df_neg['Plot'].tolist(), df_neg['scores'].tolist()],
                             'feedback': [df_complete, df_pos, feedback_df, request['threshold']]
                             # [feedback_df, df_pos]
                             })

    print(f"\nNew carrier:  {feedback_carrier}")
    print(f"\nPOST RESPONSE.\n{json.dumps(parsed, indent=4)}\n")

    return {'data':
                {'classifier': str(classifier),
                 'category': category,
                 'output': parsed,
                 'message': "Sucess!"}
            }


@app.get("/api/zero-shot/inference/")
async def get_inference_result(id_inf: int, opc: LabelsFilter):
    print("\n@ GET get_inference_result.")
    print(f"options: {opc}")
    for ix, i in enumerate(feedback_carrier):
        print(f'feedback_carrier: {ix}')
        if ix + 1 == id_inf:
            result_0 = i['feedback'][0]  # df_complete
            threshold = float(i['feedback'][3])

            print(f"threshold: {threshold}")
            print(f"sample of score: {result_0.iloc[0, 4]}, {type(result_0.iloc[0, 4])}")

            result_0['binary_label'] = result_0['scores'].apply(lambda s: float(s[0]) >= threshold / 100.0)
            print(f"binary labels: {result_0['binary_label'].tolist()}")

            if opc == LabelsFilter.pos:
                result_0 = result_0[result_0['binary_label'].eq(True)]
                print(f"positives: \n{result_0}\n")
            elif opc == LabelsFilter.neg:
                result_0 = result_0[result_0['binary_label'].eq(False)]
                print(f"negatives: \n{result_0}\n")

            # get strings of list features for printing
            result_0['labels'] = result_0['labels'].apply(lambda s: s[0])
            result_0['scores'] = result_0['scores'].apply(lambda s: s[0])

            print(f"\ncurrent state of df_complete (result of inference): \n{result_0}\n")
            result_0 = result_0.to_json(orient="index")
            parsed_0 = json.loads(result_0)
            return {'data': parsed_0,
                    'message': "Available data for feedback"}

    return {'data': [],
            'message': "Available data for feedback"}


@app.put("/api/zero-shot/feedback/{id_feed}")
async def add_training_data(id_feed: int, training_data: TrainingData):
    print("\n@ PUT add_training_data.")
    print(f"new feedback data for classifier id: {id_feed}\n")
    request = training_data.dict()
    for pair in request['pairs']:
        new_sample = pair[0]
        sample_label = pair[1]

        print("text: ", new_sample)
        # print(f"len(feedback_carrier): \n{len(feedback_carrier)}")
        for ix, i in enumerate(feedback_carrier):  # because each id is a number
            print(f"feedback_carrier current id: {i['id']}  ({i['id'].replace('~', '')} == {id_feed})")

            if int(i['id'].replace("~", "")) == id_feed:
                print(f"\nidentified in feedback carrier!")

                # single inference db
                # if '~' in id_feed:
                #
                # """ this is how single inference looks
                # feedback_carrier.append({'id': str(id_) + '~',
                #                          'threshold': request['threshold'],
                #                          'paragraphs': [{'text': request['text'],
                #                                          'decision': decision,
                #                                          'confidence': score_
                #                                          }]
                #                          })
                # """

                # dataframes
                target_pl = i['positives']
                target_fdf = i['feedback'][0]  # complete
                target_pdf = i['feedback'][1]  # pos
                target_ffdf = i['feedback'][2]  # feedback

                print(f"\ndf_complete.columns: {target_fdf.columns}\ndf_pos.columns: {target_pdf.columns}\ndf_feedback"
                      f".columns: {target_ffdf.columns}\n")

                for j in db:
                    if j['cl_id'] == id_feed:
                        print(f"identified in db!")
                        category = j['category']

                        target_pl[0].append(new_sample)  # Plot
                        target_pl[1].append(sample_label)  # scores
                        i['positives'] = target_pl

                        new_ = pd.DataFrame([['- Inserted -', new_sample, [category], [sample_label]]],
                                            columns=['Author', 'Plot', 'labels', 'scores'])
                        print(f"\nnew sample:\n{new_}")

                        # UPDATE DB
                        if len(j['data']) == 0:
                            j['data'] = new_
                        else:
                            j['data'] = pd.concat([j['data'], new_], axis=0, ignore_index=True)

                        print(f"\nnew ['data'] for {id_feed}:\n{j['data']}\n")

                        target_fdf = pd.concat([new_, target_fdf], axis=0).reset_index(drop=True)
                        target_ffdf = pd.concat([new_, target_ffdf], axis=0).reset_index(drop=True)
                        target_pdf = pd.concat([new_, target_pdf], axis=0).reset_index(drop=True)

                        # UPDATE COMPLETE and POS and FEEDBACK
                        # i['feedback'][0] = target_fdf  # df_complete  # do not update df_complete to show inference always the same
                        i['feedback'][1] = target_pdf  # df_pos
                        i['feedback'][2] = target_ffdf  # df_feedback
                        feedback_carrier[ix] = i
                        break

            if '~' in i['id']:
                break

    return {'message': f"New sample update! classifer_id: {id_feed}"}


@app.delete("/api/zero-shot/feedback/{index}")
async def correct_inferece_sample(id_feed: int, target_index: int):
    print("\n@ DELETE request_delete_positive.")
    print(f"searching index {target_index} for deletion...")

    for ixx, i in enumerate(feedback_carrier):  # because each id is a number
        if i['id'] == id_feed:
            print(f"\nidentified in feedback carrier!")
            target_pl = i['positives']
            target_fdf = i['feedback'][0]  # complete
            target_pdf = i['feedback'][1]  # pos
            target_ffdf = i['feedback'][2]  # feedback

            print(f"\ndf_complete.columns: {target_fdf.columns}\ndf_pos.columns: {target_pdf.columns}\ndf_feedback"
                  f".columns: {target_ffdf.columns}\n")

            for j in db:
                if j['cl_id'] == id_feed:
                    print(f"identified in db!")
                    category = j['category']

                    target_plot = target_fdf.loc[target_index, 'Plot']
                    print(f"\n*** target_plot for deletion : {target_plot} ***\n")

                    new_ = pd.DataFrame([['- Inserted -', target_plot, [category], [0.0]]],
                                        columns=['Author', 'Plot', 'labels', 'scores'])
                    print(f"\nnew sample:\n{new_}")

                    # UPDATE DB
                    if len(j['data']) == 0:
                        j['data'] = new_
                    else:
                        j['data'] = pd.concat([j['data'], new_], axis=0, ignore_index=True)

                    print(f"\nnew ['data']:\n{j['data']}")
                    feedback_carrier[ixx] = i
                    print("New sample added (negative) to 'data' in db!")
                    return {'message': "New sample added (negative) to 'data' in db!"}

            for ix, plot_p in zip(list(target_pdf.index), target_pdf['Plot'].tolist()):
                target_plot = target_fdf.loc[target_index, 'Plot']  # get plot in feedback
                print("found on fdf!")
                print(f"\nstatic target_plot (in feedback_df): \n{target_plot}")
                print(f"dynamic checker plot (in pos_df): \n{plot_p}\n")
                if plot_p == target_plot:
                    print(f"found on pdf! removing data from index: {ix}")
                    # DELETION
                    target_pdf.drop(index=ix, inplace=True)
                    target_fdf.drop(index=target_index, inplace=True)
                    target_pl[0].remove(target_plot)
                    return {'message': "Sample update (deletion)!"}
                else:
                    print("Data wasn't found in positives_df...")
                    return {
                        'message': "Couldn't delete data; it wasn't found on the current database of positive samples!"}
        print("Data wasn't found in feedback_df...")
        return {'message': "Couldn't delete data; it wasn't found on the current database (feedback)"}
    return {'message': "Feedback data is empty. First run inference against your data!"}


@app.get("/api/zero-shot/feedback/")
async def get_feedback_data(id_feed: int):
    print("\n@ GET feedback_data.")

    db_ = {i: db[i] for i in range(len(db))}
    print(f"db_:{db_}, len(db_):{len(db_)}, len(db):{len(db)}")

    if len(db) > 0:
        out_dt = {k: dict() for k in range(len(db))}
        for db_i, value_dt in zip(db, out_dt.values()):
            update_dt = {k: v for k, v in db_i.items()}
            update_dt['model'] = update_dt['model'].task

            if len(update_dt['data']) > 0:  # parse df inside data
                update_dt['data'] = update_dt['data'].to_json(orient="records")

            print(f"model.task: {update_dt['model']}")
            value_dt.update(update_dt)

    # dump_this is for printing on terminal and when len(db)==0 --> return db_ which is pretty much just an empty dict
    dump_this = json.dumps(db_, indent=4) if len(db) == 0 else json.dumps(out_dt, sort_keys=True, indent=4,
                                                                          ensure_ascii=False)
    print(f"\ndump_this: {dump_this}")

    # creates dictionary with db for cl_id, model and data --> ky; values are list of elements from classifiers
    if len(db) > 0:
        dt = {k: [] for k in db[0]}
        for ky in db[0].keys():
            for i in db:
                if ky != "model":
                    dt[ky].append(i[ky])
                elif ky == "model":
                    dt[ky].append(i[ky].task)

        # covert to dataframe because its easier to format to json string
        print(f"\ndt: {dt}")
        df_dt = pd.DataFrame.from_dict(dt, orient="index")
        df_dt = df_dt.transpose()
        print(f"\ndf_dt: \n{df_dt}")

        result = df_dt.iloc[id_feed - 1, :]
        result = result.to_json(orient="records")
        result = json.loads(result)
        print(f"\nresult (index for df): \n{result}\n")

    return {'data':
                {'models': 'list of available models',
                 'db': result if len(db) > 0 else dump_this},
            'message': "HuggingFace model-hub | current models"
            }


@app.post("/api/zero-shot/retrain/{data_id}")
async def retrain_classifier(data_id: int):
    # commented code works for automatic custom size dataset generation...
    print(f"\n@ POST request_retrain; data_id: {data_id}")

    # this will get a stored pipeline under id and init model var with it
    for i in db:
        stored_pipeline = None
        if i['cl_id'] == data_id:
            df_retrain = i['data']
            df_retrain.loc[:, 'labels'] = df_retrain['labels'].apply(lambda s: s[0])
            df_retrain.loc[:, 'scores'] = df_retrain['scores'].apply(lambda s: s[0])
            stored_pipeline = i['model']  # get the pipeline
            try:
                description = i['description']
                print(f"description found: {description}")
            except KeyError:
                pass

    model = stored_pipeline
    print(f"\ndf_retrain: \n{df_retrain}")
    labels_list = df_retrain['labels'].tolist()
    print(f"labels found: {labels_list}")
    label_nm = list(set(labels_list))[0]

    df_retrain['labels'] = df_retrain['scores']

    X = df_retrain['Plot'].tolist()
    y = df_retrain['labels'].tolist()
    df_retrain.drop(columns=["Author", "scores"], inplace=True)

    # print(f"df_retrain (edit2): \n{df_retrain}")
    print(f"X: {X}\ny: {y}")

    # try except when number of samples is less than 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    data_dictionary = {'Xtrain': X_train, "ytrain": y_train, "Xtest": X_test, "ytest": y_test, "Xval": X_val,
                       "yval": y_val}

    # TOKENIZATION
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    try:
        if description == "fine-tuned model":
            print("using tuned tokenizer")
            tokenizer = BartTokenizer.from_pretrained('results/fine-tuning/',
                                                      vocab_file="results/fine-tuning/vocab.json",
                                                      merges_file="results/fine-tuning/merges.txt")
    except UnboundLocalError:
        print("There no assign description for this classifier...")
    finally:
        print("BartTokenizer ready!")

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    val_encodings = tokenizer(X_val, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    # PYTORCH OBJECTS
    train_dataset = FeedbackDataset(train_encodings, y_train)
    val_dataset = FeedbackDataset(val_encodings, y_val)
    test_dataset = FeedbackDataset(test_encodings, y_test)

    # FINE-TUNING
    # Option 1: FINE-TUNING WITH TRAINER
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=2,  # 10... total number of training epochs
        per_device_train_batch_size=4,  # 16 ... batch size per device during training
        per_device_eval_batch_size=8,  # 64 ... batch size for evaluation
        warmup_steps=1,  # 500 ... number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1,
    )

    # model here should be whatever classifier is in this id and not always bart-large-mnli, it will only work for demo
    try:
        if description == "fine-tuned model":
            print("using tuned model...")
            model = BartForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path="C:/Users/lavml/Documents/SoSe20/nlp/BERT/restapi/results/fine-tuning"
                                              "/pytorch_model.bin", local_files_only=True)

    except UnboundLocalError:
        print("There no assign description for this classifier...")
        model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

    finally:
        print("BartForSequenceClassification ready!")

    try:
        if description == "fine-tuned model":
            print("using tuned tokenizer")
            tokenizer = BartTokenizer.from_pretrained('results/fine-tuning/',
                                                      vocab_file="results/fine-tuning/vocab.json",
                                                      merges_file="results/fine-tuning/merges.txt")
    except UnboundLocalError:
        print("There no assign description for this classifier...")
    finally:
        print("BartTokenizer ready!")

    try:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset  # evaluation dataset
        )
    except RuntimeError:
        print("CUDA RuntimeError. Device changed to cpu")
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            no_cuda=True,
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset  # evaluation dataset
        )

    print("\ntraining...")

    trainer.train()

    try:
        trainer.save_model('results/trainer/')

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained('results/fine-tuning/')
        tokenizer.save_pretrained('results/fine-tuning/')

    except:
        print("error saving with results/[trainer, fine-tuning]")
        pass

    print("fine-tuned and stored, output_dir = './results/fine-tuning/'")  #

    # LOAD MODEL TO DB
    pretrained_model = BartForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="C:/Users/lavml/Documents/SoSe20/nlp/BERT/restapi/results/fine-tuning/", local_files_only=True)
    tokenizer = BartTokenizer.from_pretrained('results/fine-tuning/', vocab_file="results/fine-tuning/vocab.json",
                                              merges_file="results/fine-tuning/merges.txt")

    classifier = pipeline('zero-shot-classification', model=pretrained_model, tokenizer=tokenizer)  # , device=0)

    # add it to the db

    new_id = 1 if len(db) == 0 else db[-1]['cl_id'] + 1
    db.append(
        {'cl_id': new_id, 'model': classifier, 'category': label_nm, 'data': [], 'description': 'fine-tuned model'})

    return {'data': None,
            'message': f"Fine-tuned Model apended to the database under the id: {new_id}. Try the new resource here "
                       f"--> http://127.0.0.1:8000/api/zero-shot/{new_id}"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# todo:
#       choose dataset for re-train and create enum for the model_name only for zero shot to start; include name in db.
#       create results/fine-tuning and trainer dirs programmatically
