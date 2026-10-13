"""Final MedTool-Fine research implementation."""
import copy

class AnswerSession:
    """Only a precompiled patient-answer transition table; no reference tools/parameters."""

    def __init__(self, record, aliases):
        self.record = record
        self.state = record['initial_state']
        self.aliases = aliases
        self.answered = {}

    def answer(self, concept):
        concept = self.aliases.get(concept.strip().lower(), concept.strip().lower())
        if concept in self.answered:
            text = self.answered[concept]
            kind = 'REPEATED_KNOWN_ANSWER'
        else:
            transition = self.record['states'][self.state].get(concept)
            if transition:
                text = transition['text']
                self.state = transition['next_state']
                kind = 'CONTROLLED_NEW_INFORMATION'
            else:
                text = 'No additional information is available for this question.'
                kind = 'NO_INFORMATION'
            self.answered[concept] = text
        return {'concept': concept, 'text': text, 'kind': kind}

class _BatchAnswerSession(AnswerSession):

    def answer_batch(self, concepts):
        names = [self.aliases.get(c.strip().lower(), c.strip().lower()) for c in concepts]
        names = list(dict.fromkeys(names))
        initial = self.record['states'][self.state]
        unavailable = sorted((c for c in names if c not in initial or c in self.answered))
        available = sorted((c for c in names if c in initial and c not in self.answered))

        def path(state, remaining):
            if not remaining:
                return []
            for c in remaining:
                tr = self.record['states'][state].get(c)
                if tr and tr['text'] == initial[c]['text']:
                    tail = path(tr['next_state'], [n for n in remaining if n != c])
                    if tail is not None:
                        return [c] + tail
            return None
        available_order = path(self.state, available)
        if available_order is None:
            raise ValueError('BATCH_ANSWER_CONFLICT')
        order = unavailable + available_order
        result = []
        for name in order:
            if name in initial and name not in self.answered:
                current = self.record['states'][self.state].get(name)
                if current is None or current['text'] != initial[name]['text']:
                    raise ValueError('BATCH_ANSWER_CONFLICT')
            result.append(self.answer(name))
        return result

LEGACY_NO_MATCH = 'No additional information is available for this question.'

PUBLIC_NO_MATCH = 'NO_MATCHING_ANSWER: This question did not match an answer in the current information state. This is not evidence that the clinical fact is absent or impossible to obtain. Review the supplied record, prior answers and the question vocabulary.'

def question_aliases(aliases):
    groups = {}
    for name, target in aliases.items():
        groups.setdefault(' '.join(name.strip().lower().replace('_', ' ').split()), set()).add(target)
    expanded = dict(aliases)
    for name, targets in groups.items():
        if len(targets) == 1:
            for variant in (name, name.replace(' ', '_')):
                expanded.setdefault(variant, next(iter(targets)))
    return expanded

def public_answers(events):
    result = copy.deepcopy(events)
    for event in result:
        if event['kind'] == 'NO_INFORMATION' or (event['kind'] == 'REPEATED_KNOWN_ANSWER' and event['text'] == LEGACY_NO_MATCH):
            event['text'] = PUBLIC_NO_MATCH
    return result

class BatchAnswerSession(_BatchAnswerSession):

    def __init__(self, record, aliases):
        super().__init__(record, question_aliases(aliases))

