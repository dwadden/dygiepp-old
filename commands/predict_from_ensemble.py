#! /usr/bin/env python

from sys import argv
import os
from os import path
import itertools
import numpy as np
import json
from collections import Counter

import torch

from allennlp.data import Vocabulary

from dygie.models.shared import fields_to_batches
from dygie.commands import predict_dygie as pdy


def unwrap(trig_pred_dict):
    res = {}
    for k in sorted(trig_pred_dict.keys()):
        v = trig_pred_dict[k]
        for doc_key, doc_vals in v.items():
            if doc_key in res:
                res[doc_key].append(doc_vals)
            else:
                res[doc_key] = [doc_vals]
    return res


def modal_prediction(predictions):
    counts = Counter(predictions)
    most_common = counts.most_common()[0][0]
    return most_common


def decode_trigger(outputs, trigger_vocabs):
    sentence_lengths = [entry["sentence_lengths"].item() for entry in outputs]
    assert len(set(sentence_lengths)) == 1
    sentence_lengths = sentence_lengths[0]
    # scores = torch.cat([output["trigger_scores"].unsqueeze(-1) for output in outputs], dim=-1)
    # mean_scores, _ = scores.median(dim=-1)
    # predicted_triggers = mean_scores.argmax(dim=1)
    trigger_dict = {}
    for i in range(sentence_lengths):
        the_predictions = []
        for output, vocab in zip(outputs, trigger_vocabs):
            trig_label = output["predicted_triggers"][i].item()
            the_predictions.append(vocab.get_token_from_index(trig_label, namespace="trigger_labels"))
        the_prediction = modal_prediction(the_predictions)
        if the_prediction != "":
            trigger_dict[i] = the_prediction

    return trigger_dict


def decode_one_argument(output, decoded_trig, vocab):
    argument_dict = {}
    argument_dict_with_scores = {}
    the_highest, _ = output["argument_scores"].max(dim=0)
    for i, j in itertools.product(range(output["num_triggers_kept"]),
                                  range(output["num_argument_spans_kept"])):
        trig_ix = output["top_trigger_indices"][i].item()
        arg_span = tuple(output["top_argument_spans"][j].tolist())
        arg_label = output["predicted_arguments"][i, j].item()
        # Only include the argument if its putative trigger is predicted as a real trigger.
        if arg_label >= 0 and trig_ix in decoded_trig:
            arg_score = output["argument_scores"][i, j, arg_label + 1].item()
            label_name = vocab.get_token_from_index(arg_label, namespace="argument_labels")
            argument_dict[(trig_ix, arg_span)] = label_name
            # Keep around a version with the predicted labels and their scores, for debugging
            # purposes.
            argument_dict_with_scores[(trig_ix, arg_span)] = (label_name, arg_score)

    return argument_dict, argument_dict_with_scores


def majority_vote(argument_dicts, argument_dicts_with_scores):
    # Keep an argument if at least 2 folds had it.
    counts = Counter()
    argument_tupss = [[(k, v) for k, v in entry.items()] for entry in argument_dicts]

    argument_scores = [[(k, v) for k, v in entry.items()] for entry in argument_dicts_with_scores]

    for arg_tups in argument_tupss:
        for arg_tup in arg_tups:
            counts[arg_tup] += 1

    keep = [k for k, v in counts.items() if v > 1]
    argument_dict = dict(keep)

    # Just set all the scores to 1 to get this done.
    argument_dict_with_scores = {k: (v, 1.0) for k, v in argument_dict.items()}

    return argument_dict, argument_dict_with_scores


def decode_arguments(outputs, decoded_trig, vocabs):
    decoded = [decode_one_argument(output, decoded_trig, vocab)
               for output, vocab in zip(outputs, vocabs)]

    argument_dicts = [x[0] for x in decoded]
    argument_dicts_with_scores = [x[1] for x in decoded]

    if not decoded_trig:
        return decoded[0]
    else:
        argument_dict, argument_dict_with_scores = majority_vote(argument_dicts,
                                                                 argument_dicts_with_scores)
        return argument_dict, argument_dict_with_scores


def decode(trig_list, arg_list, trigger_vocabs, arg_vocabs):
    """
    Largely copy-pasted from what happens in dygie.
    """
    ignore = ["loss", "decoded_events"]
    trigss = [fields_to_batches({k: v.detach().cpu() for k, v in entry.items() if k not in ignore})
             for entry in trig_list]
    trigss = list(zip(*trigss))

    argss = [fields_to_batches({k: v.detach().cpu() for k, v in entry.items() if k not in ignore})
            for entry in arg_list]
    argss = list(zip(*argss))

    res = []

    # Collect predictions for each sentence in minibatch.
    for trigs, args in zip(trigss, argss):
        decoded_trig = decode_trigger(trigs, trigger_vocabs)
        decoded_args, decoded_args_with_scores = decode_arguments(args, decoded_trig, arg_vocabs)
        entry = dict(trigger_dict=decoded_trig, argument_dict=decoded_args,
                     argument_dict_with_scores=decoded_args_with_scores)
        res.append(entry)

    return res


def get_pred_dicts(pred_dir):
    res = {}
    for name in os.listdir(pred_dir):
        doc_key = name.replace(".th", "")
        res[doc_key] = torch.load(path.join(pred_dir, name))

    return res


def predict_one(trig, arg, gold, trigger_vocabs, arg_vocabs):
    sentence_lengths = [len(entry) for entry in gold["sentences"]]
    sentence_starts = np.cumsum(sentence_lengths)
    sentence_starts = np.roll(sentence_starts, 1)
    sentence_starts[0] = 0

    decoded = decode([x["events"] for x in trig], [x["events"] for x in arg],
                     trigger_vocabs, arg_vocabs)
    cleaned = pdy.cleanup_event(decoded, sentence_starts)
    res = {}
    res.update(gold)
    res["predicted_events"] = cleaned
    pdy.check_lengths(res)
    encoded = json.dumps(res, default=int)
    return encoded


def get_gold_data(test_file):
    data = pdy.load_json(test_file)
    res = {x["doc_key"]: x for x in data}
    return res


def predict_from_ensemble(trigger_prediction_dirs, trigger_vocab_dirs, arg_prediction_dirs,
                          arg_vocab_dirs, test_file, output_file):
    trig_pred_dict = {k: get_pred_dicts(v) for k, v in trigger_prediction_dirs.items()}
    trigger_vocabs = [Vocabulary.from_files(v) for v in trigger_vocab_dirs]

    arg_pred_dict = {k: get_pred_dicts(v) for k, v in arg_prediction_dirs.items()}
    arg_vocabs = [Vocabulary.from_files(v) for v in arg_vocab_dirs]

    gold = get_gold_data(test_file)

    for arg_pred_entry in arg_pred_dict.values():
        assert all([set(arg_pred_entry.keys()) == set(entry.keys()) for entry in trig_pred_dict.values()])

    trig_preds = unwrap(trig_pred_dict)
    arg_preds = unwrap(arg_pred_dict)
    with open(output_file, "w") as f:
        for doc in trig_preds:
            one_pred = predict_one(trig_preds[doc], arg_preds[doc], gold[doc], trigger_vocabs, arg_vocabs)
            f.write(one_pred + "\n")


def main():
    ensemble_dir = argv[1]
    out_dir = argv[2]
    trigger_seed = int(argv[3])

    trigger_dirs = {k: f"{ensemble_dir}/triger_ensemble_seed_{trigger_seed}_{k}/prediction_scores_test"
                    for k in [0, 1, 2, 3]}
    trigger_vocab_dirs = [f"{ensemble_dir}/triger_ensemble_seed_{trigger_seed}_{k}/vocabulary"
                          for k in [0, 1, 2, 3]]

    arg_dirs = {k: f"{ensemble_dir}/args_ensemble_{k}/prediction_scores_test"
                for k in [0, 1, 2, 3]}
    arg_vocab_dirs = [f"{ensemble_dir}/args_ensemble_{k}/vocabulary"
                      for k in [0, 1, 2, 3]]

    test_file = "/data/dwadden/proj/dygie/dygie-experiments/datasets/ace-event-tongtao-settings/json/test.json"
    output_file = path.join(out_dir, f"arg_predictions_{trigger_seed}.json")
    predict_from_ensemble(trigger_dirs, trigger_vocab_dirs, arg_dirs, arg_vocab_dirs, test_file, output_file)


if __name__ == '__main__':
    main()
