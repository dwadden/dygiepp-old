from collections import namedtuple
import json
from dygie.models.shared import fields_to_batches
import numpy as np


CorefPrediction = namedtuple("CorefPrediction", ["entity", "antecedent"])


def get_sentence_of_span(span, sentence_starts, doc_tokens):
    """
    Return the index of the sentence that the span is part of.
    """
    # Inclusive sentence ends
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [span[0] >= start and span[1] <= end
                  for start, end in zip(sentence_starts, sentence_ends)]
    assert sum(in_between) == 1
    the_sentence = in_between.index(True)
    return the_sentence


class Dataset:
    def __init__(self, json_file):
        with open(json_file) as f:
            self.js = [json.loads(line) for line in f]
        self.documents = [Document(js) for js in self.js]

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)


class Document:
    def __init__(self, js):
        self._doc_key = js["doc_key"]
        entries = fields_to_batches(js, ["doc_key", "clusters", "predicted_clusters", "top_coref"])
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        self.sentence_starts = sentence_starts
        self.sentences = [Sentence(entry, sentence_start, sentence_ix)
                          for sentence_ix, (entry, sentence_start)
                          in enumerate(zip(entries, sentence_starts))]
        if "clusters" in js:
            self.top_coref = self._parse_top_coref(js["top_coref"]) if "top_coref" in js else None
            self.clusters = [Cluster(entry, i, self)
                             for i, entry in enumerate(js["clusters"])]
        if "predicted_clusters" in js:
            self.predicted_clusters = [Cluster(entry, i, self)
                                       for i, entry in enumerate(js["predicted_clusters"])]

    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def get_sentence_of(self, span):
        return get_sentence_of_span((span.start_doc, span.end_doc),
                                    self.sentence_starts, self.n_tokens)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))

    def _parse_top_coref(self, top_coref):
        res = []
        for entity, antecedent in zip(top_coref["top_spans"], top_coref["top_antecedents"]):
            def make_span(pair):
                sentence_ix = get_sentence_of_span(pair, self.sentence_starts, self.n_tokens)
                sentence = self[sentence_ix]
                span = Span(pair[0], pair[1], sentence.text, sentence.sentence_start)
                return span

            span_entity = make_span(entity)
            span_antecedent = make_span(antecedent)
            res.append(CorefPrediction(span_entity, span_antecedent))

        return res



    def find_cluster(self, entity, predicted=True):
        """
        Search through coreference clusters and return the one containing the query entity, if it's
        part of a cluster. If we don't find a match, return None.
        """
        clusters = self.predicted_clusters if predicted else self.clusters
        for clust in clusters:
            for entry in clust:
                if entry.span == entity.span:
                    return clust

        return None

    def find_predicted_coref(self, entity, exact=False):
        """
        Search through `top_coref` and see if the entity got predicted as part of a cluster.
        """
        matches = []
        for coref in self.top_coref:
            pred_ent = coref[0]
            if pred_ent.compare(entity.span, exact=exact):
                matches.append(coref)

        return matches

    def find_predicted_referred(self, entity, exact=False):
        """
        Search through `top_coref` to find spans whose antecedent is the current entity.
        """
        res = []

        for coref in self.top_coref:
            pred_ante = coref[1]
            if pred_ante.compare(entity.span, exact=exact):
                res.append(coref)

        return res

    def get_entity_matches(self, span, exact=False):
        """
        Is this span an entity in the document?
        """
        all_entities = [x.ner for x in self.sentences]
        all_entities = [x for y in all_entities for x in y]

        matches = []

        for guess in all_entities:
            if span.compare(guess.span, exact=exact):
                matches.append(guess)

        return matches

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix):
        self.sentence_start = sentence_start
        self.text = entry["sentences"]
        self.sentence_ix = sentence_ix
        # Gold
        if "ner_flavor" in entry:
            self.ner = [NER(this_ner, self.text, sentence_start, flavor=this_flavor)
                        for this_ner, this_flavor in zip(entry["ner"], entry["ner_flavor"])]
        else:
            self.ner = [NER(this_ner, self.text, sentence_start)
                        for this_ner in entry["ner"]]
        self.relations = [Relation(this_relation, self.text, sentence_start) for
                          this_relation in entry["relations"]]
        if "events" in entry:
            self.events = Events(entry["events"], self.text, sentence_start)

        # Predicted
        if "predicted_ner" in entry:
            self.predicted_ner = [NER(this_ner, self.text, sentence_start, flavor=None) for
                                  this_ner in entry["predicted_ner"]]
        if "predicted_relations" in entry:
            self.predicted_relations = [Relation(this_relation, self.text, sentence_start) for
                                        this_relation in entry["predicted_relations"]]
        if "predicted_events" in entry:
            self.predicted_events = Events(entry["predicted_events"], self.text, sentence_start)

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)

    def get_flavor(self, argument):
        the_ner = [x for x in self.ner if x.span == argument.span]
        if len(the_ner) > 1:
            print("Weird")
        if the_ner:
            the_flavor = the_ner[0].flavor
        else:
            the_flavor = None
        return the_flavor


class Span:
    def __init__(self, start, end, text, sentence_start):
        self.start_doc = start
        self.end_doc = end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = start - sentence_start
        self.end_sent = end - sentence_start
        self.span_sent = (self.start_sent, self.end_sent)
        self.text = text[self.start_sent:self.end_sent + 1]

    def overlaps(self, other):
        return (self in other) or (other in self)

    def compare(self, other, exact=False):
        if exact:
            return self == other
        else:
            return self.overlaps(other)

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.text == other.text)

    def __hash__(self):
        tup = self.span_doc + self.span_sent + (" ".join(self.text),)
        return hash(tup)

    def __contains__(self, other):
        # True if `other` is in `self`.
        return (self.start_doc <= other.start_doc) and (self.end_doc >= other.end_doc)


class Token:
    def __init__(self, ix, text, sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))


class Trigger:
    def __init__(self, token, label):
        self.token = token
        self.label = label

    def __repr__(self):
        return self.token.__repr__()[:-1] + ", " + self.label + ")"


class Argument:
    def __init__(self, span, role, event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return self.span.__repr__()[:-1] + ", " + self.event_type + ", " + self.role + ")"

    def __eq__(self, other):
        return (self.span == other.span and
                self.role == other.role and
                self.event_type == other.event_type)

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))


class NER:
    def __init__(self, ner, text, sentence_start, flavor=None):
        self.span = Span(ner[0], ner[1], text, sentence_start)
        self.label = ner[2]
        self.flavor = flavor

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label and
                self.flavor == other.flavor)

class Relation:
    def __init__(self, relation, text, sentence_start):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1, end1, text, sentence_start)
        span2 = Span(start2, end2, text, sentence_start)
        self.pair = (span1, span2)
        self.label = label

    def __repr__(self):
        return self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label


class Event:
    def __init__(self, event, text, sentence_start):
        trig = event[0]
        args = event[1:]
        trigger_token = Token(trig[0], text, sentence_start)
        self.trigger = Trigger(trigger_token, trig[1])

        self.arguments = []
        for arg in args:
            span = Span(arg[0], arg[1], text, sentence_start)
            self.arguments.append(Argument(span, arg[2], self.trigger.label))

    def __repr__(self):
        res = "<"
        res += self.trigger.__repr__() + ":\n"
        for arg in self.arguments:
            res += 6 * " " + arg.__repr__() + ";\n"
        res = res[:-2] + ">"
        return res


class Events:
    def __init__(self, events_json, text, sentence_start):
        self.event_list = [Event(this_event, text, sentence_start) for this_event in events_json]
        self.triggers = set([event.trigger for event in self.event_list])
        self.arguments = set([arg for event in self.event_list for arg in event.arguments])

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, i):
       return self.event_list[i]

    def __repr__(self):
        return "\n\n".join([event.__repr__() for event in self.event_list])

    def span_matches(self, argument):
        return set([candidate for candidate in self.arguments
                    if candidate.span.span_sent == argument.span.span_sent])

    def event_type_matches(self, argument):
        return set([candidate for candidate in self.span_matches(argument)
                    if candidate.event_type == argument.event_type])

    def matches_except_event_type(self, argument):
        matched = [candidate for candidate in self.span_matches(argument)
                   if candidate.event_type != argument.event_type
                   and candidate.role == argument.role]
        return set(matched)

    def exact_match(self, argument):
        for candidate in self.arguments:
            if candidate == argument:
                return True
        return False


class Cluster:
    def __init__(self, cluster, cluster_id, document):
        members = []
        for entry in cluster:
            sentence_ix = get_sentence_of_span(entry, document.sentence_starts, document.n_tokens)
            sentence = document[sentence_ix]
            span = Span(entry[0], entry[1], sentence.text, sentence.sentence_start)
            ners = [x for x in sentence.ner if x.span == span]
            assert len(ners) <= 1
            ner = ners[0] if len(ners) == 1 else None
            to_append = ClusterMember(span, ner, sentence, cluster_id)
            members.append(to_append)

        self.members = members
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"{self.cluster_id}: " + self.members.__repr__()

    def __getitem__(self, ix):
        return self.members[ix]


class ClusterMember:
    def __init__(self, span, ner, sentence, cluster_id):
        self.span = span
        self.ner = ner
        self.sentence = sentence
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"<{self.sentence.sentence_ix}> " + self.span.__repr__()
