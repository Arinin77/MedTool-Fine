"""Final MedTool-Fine research implementation."""
from copy import deepcopy
import re
import math
import unicodedata

def normalize(text):
    """Unicode/case/spacing/punctuation only; no fuzzy or semantic matching."""
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join((c for c in text if c.isalnum()))

UNIT_ALIASES = {'percent': ['%', 'percent', 'percentage', 'percentage points', 'pct', 'percent (%)', 'percentage (%)'], 'proportion': ['proportion', 'probability', 'fraction', '0-1', '0 to 1'], 'points': ['', 'point', 'points', 'score', 'score points', 'integer score', 'dimensionless', 'unitless', 'none', 'n/a'], 'dimensionless': ['', 'dimensionless', 'unitless', 'index', 'index value', 'score', 'points', 'none', 'n/a'], 'count': ['', 'count', 'counts', 'criteria', 'criteria met', 'items', 'number'], 'category': ['', 'category', 'class', 'grade', 'risk category', 'none', 'n/a'], 'boolean': ['', 'boolean', 'bool', 'none', 'n/a'], 'mL/min/1.73m2': ['mL/min/1.73m2', 'mL/min/1.73m^2', 'mL/min/1.73m²', 'mL/min per 1.73m2', 'mL/min per 1.73m^2', 'mL min-1 1.73m-2'], 'mL/min': ['mL/min', 'mL per min', 'mL/minute', 'mL min-1'], 'years': ['year', 'years', 'yr', 'yrs']}

def unit_factor(actual_unit, canonical_unit):

    def norm_unit(s):
        return normalize(s.replace('%', 'percent'))
    key = norm_unit(actual_unit)
    if canonical_unit == 'percent':
        if key in {norm_unit(s) for s in UNIT_ALIASES['percent']}:
            return 1.0
        if key in {norm_unit(s) for s in UNIT_ALIASES['proportion']}:
            return 100.0
        return None
    if key in {norm_unit(s) for s in UNIT_ALIASES.get(canonical_unit, [])}:
        return 1.0
    return None

def compare_value(value, unit, field):
    """(display precision match, original-tolerance projected match, error)."""
    factor = unit_factor(unit, field['unit'])
    if factor is None:
        return (False, False, 'UNIT_UNMATCHED')
    expected, kind = (field['expected'], field['kind'])
    if kind == 'boolean':
        ok = type(value) is bool and value == expected
        return (ok, ok, None if ok else 'VALUE_MISMATCH')
    if kind == 'category':
        aliases = field.get('value_aliases', {})
        ok = isinstance(value, str) and aliases.get(normalize(value)) == normalize(expected)
        return (ok, ok, None if ok else 'CATEGORY_UNMATCHED')
    if type(value) not in (int, float) or not math.isfinite(value):
        return (False, False, 'NUMERIC_TYPE_MISMATCH')
    actual = value * factor
    if 'numeric_bounds' in field:
        low, high = field['numeric_bounds']
        if not low <= actual <= high:
            return (False, False, 'OUT_OF_OUTPUT_RANGE')
    if kind == 'integer':
        ok = actual == expected
        return (ok, ok, None if ok else 'VALUE_MISMATCH')
    tol = field['tolerance']
    strict = math.isclose(actual, expected, abs_tol=tol['abs_tol'], rel_tol=tol['rel_tol'])
    display_tol = 0.5 * 10 ** (math.floor(math.log10(abs(expected))) - 2) if expected else 0
    display_tol = field.get('display_tolerance_override', display_tol)
    display = strict or abs(actual - expected) <= display_tol
    return (display, strict, None if display else 'VALUE_MISMATCH')

NUM = '[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?'

RENAL = re.compile('(?<![\\w.])(?P<n>' + NUM + ')\\s*(?P<u>ml\\s*/\\s*min(?:ute)?(?:\\s*(?:/|per)\\s*1\\.73\\s*m\\s*\\^?2)?)', re.I)

PERCENT = re.compile('(?<![\\w.])(?P<n>' + NUM + ')\\s*(?:%|percent\\b)', re.I)

POINTS = re.compile('(?<![\\w.])(?P<n>' + NUM + ')\\s*points?\\b', re.I)

_BASE_SCORE = re.compile('\\bscore\\s*(?::|=|is|of)\\s*(?:approximately\\s+|about\\s+)?(?P<n>' + NUM + ')(?![\\w.])', re.I)

INSUFFICIENT = re.compile('\\b(?:cannot|can not|can\\x27t|could not|unable to)\\b.{0,55}\\b(?:determin\\w*|calculat\\w*|estimat\\w*|assess\\w*|answer\\w*|comput\\w*)|\\b(?:definite|definitive|exact)\\b.{0,65}\\b(?:cannot|can not|could not)\\b|\\b(?:insufficient|not enough)\\s+(?:information|data|evidence)\\b', re.I)

def canonical(text):
    return unicodedata.normalize('NFKC', text).replace('−', '-')

def method_mentions(text, vocabulary):
    """Longest overlapping public name wins; no Gold-based disambiguation."""
    matches = []
    for tid, entry in vocabulary.items():
        for label in entry['labels']:
            tokens = re.findall('[a-z0-9]+', canonical(label).lower())
            if not tokens or (len(tokens) == 1 and len(tokens[0]) < 3):
                continue
            pattern = '(?<!\\w)' + '[\\s_()\\-]*'.join(map(re.escape, tokens)) + '(?!\\w)'
            for m in re.finditer(pattern, text, re.I):
                matches.append((m.start(), m.end(), tid))
    return sorted({tid for start, end, tid in matches if not any((s <= start and e >= end and (e - s > end - start) for s, e, _ in matches))})

def lead_label(text, start):
    lead = text[:start].rstrip(' :=~')
    lead = re.split('[,;\\n]|(?<!\\d)\\.(?!\\d)', lead)[-1]
    lead = re.sub('\\b(?:is|are|approximately|about|estimated|probability|risk)\\s*$', '', lead, flags=re.I).strip()
    lead = re.sub('\\bprostate\\s+cancer\\b', 'cancer', lead, flags=re.I)
    return normalize(lead)

def _extract_base(text, vocabulary):
    text = canonical(text)
    atoms, occupied = ([], [])

    def add(m, measure, value, unit):
        start, end = m.span()
        if any((start < e and end > s for s, e in occupied)):
            return
        if re.search('[<>≤≥]\\s*$', text[max(0, start - 3):start]):
            return
        atoms.append({'measure': measure, 'value': value, 'unit': unit, 'span': [start, end], 'text': m.group(), 'lead': lead_label(text, start), 'qualified': bool(re.search('\\b(?:if|might|may|could|would|not true|cannot conclude)\\b', re.split('[;\\n]|(?<!\\d)\\.(?!\\d)', text[:start])[-1], re.I))})
        occupied.append((start, end))
    for m in RENAL.finditer(text):
        unit = 'mL/min/1.73m2' if '1.73' in m['u'] else 'mL/min'
        add(m, 'egfr' if '1.73' in m['u'] else 'renal_clearance', float(m['n']), unit)
    for pattern, measure, unit in [(PERCENT, 'probability', 'percent'), (POINTS, 'score', 'points'), (_BASE_SCORE, 'score', 'points')]:
        for m in pattern.finditer(text):
            add(m, measure, float(m['n']), unit)
    bare = re.fullmatch('\\s*(' + NUM + ')\\s*\\.?\\s*', text)
    if bare:
        add(bare, 'scalar', float(bare[1]), '')
    for m in re.finditer('\\b(?:pulmonary embolism|PE)\\s+(?P<aux>can|cannot|can not)\\s+be\\s+(?:safely\\s+)?(?:excluded|ruled out)\\b', text, re.I):
        add(m, 'pe_decision', 'PE can be excluded' if m['aux'].lower() == 'can' else 'imaging required', 'category')
    for m in re.finditer('\\b(low|moderate|intermediate|high)\\s+risk\\b', text, re.I):
        add(m, 'risk_category', m[1].lower(), 'category')
    return {'text': text, 'atoms': atoms, 'method_ids': method_mentions(text, vocabulary), 'insufficiency_statement': bool(INSUFFICIENT.search(text))}

def relevant(atom, field, required):
    aliases = set(field['aliases'])
    if atom['measure'] == 'pe_decision':
        return field['kind'] == 'category' and bool({'decision', 'pedecision', 'peruleoutdecision', 'pregnancyadaptedyearsresult'} & aliases)
    if atom['measure'] == 'risk_category':
        return field['kind'] == 'category' and bool({'riskcategory', 'riskband', 'category'} & aliases)
    if atom['measure'] == 'egfr':
        return bool({'egfr', 'estimatedgfr', 'gfr'} & aliases)
    if atom['measure'] == 'renal_clearance':
        return field['unit'] in ('mL/min', 'mL/min/1.73m2')
    if atom['measure'] == 'score':
        return field['unit'] in ('points', 'dimensionless', 'count')
    if atom['measure'] == 'probability':
        if any((a and atom['lead'].endswith(a) for a in aliases)):
            return True
        return len(required) == 1 and field['unit'] == 'percent'
    if atom['measure'] == 'scalar':
        return len(required) == 1 and field['kind'] in ('integer', 'number')
    return False

def _score_base_branch(extracted, branch):
    required = [f for f in branch['fields'] if f['required']]
    results = []
    for field in required:
        found = [a for a in extracted['atoms'] if relevant(a, field, required)]
        distinct = {(str(a['value']), a['unit']) for a in found}
        if any((a['qualified'] for a in found)):
            results.append({'key': field['key'], 'status': 'UNRESOLVED', 'reason': 'CONDITIONAL_OR_QUALIFIED_ASSERTION', 'atoms': found})
        elif len(distinct) > 1:
            results.append({'key': field['key'], 'status': 'UNRESOLVED', 'reason': 'MULTIPLE_RESULT_ASSERTIONS', 'atoms': found})
        elif not found:
            only_category = bool(extracted['atoms']) and all((a['measure'] == 'risk_category' for a in extracted['atoms']))
            status = 'INCOMPLETE' if only_category and field['kind'] in ('integer', 'number') else 'UNRESOLVED'
            results.append({'key': field['key'], 'status': status, 'reason': 'REQUIRED_RESULT_NOT_EXTRACTED'})
        else:
            atom = found[0]
            display, strict, error = compare_value(atom['value'], atom['unit'], field)
            status = 'CORRECT' if strict else 'DISPLAY_ONLY' if display else 'WRONG'
            if error in ('UNIT_UNMATCHED', 'CATEGORY_UNMATCHED'):
                if error == 'UNIT_UNMATCHED' and atom['unit'] or (isinstance(atom['value'], str) and normalize(atom['value']) in field.get('value_aliases', {})):
                    status = 'WRONG'
                else:
                    status = 'UNRESOLVED'
            results.append({'key': field['key'], 'status': status, 'reason': error, 'atom': atom, 'expected': field['expected'], 'unit': field['unit']})
    states = {f['status'] for f in results}
    if 'WRONG' in states:
        overall = 'WRONG'
    elif 'UNRESOLVED' in states:
        overall = 'UNRESOLVED'
    elif 'INCOMPLETE' in states:
        overall = 'INCOMPLETE'
    elif 'DISPLAY_ONLY' in states:
        overall = 'DISPLAY_ONLY'
    else:
        overall = 'CORRECT' if results else 'UNRESOLVED'
    return {'tool_id': branch['tool_id'], 'status': overall, 'fields': results}

def _apply_numeric_policy(field, policy):
    result = deepcopy(field)
    rule_name = field['evaluation_numeric_class']
    if rule_name in ('category', 'boolean'):
        return result
    rule = policy[rule_name]
    result['tolerance'] = {'abs_tol': max(rule['absolute_tolerance'], abs(field['expected']) * rule['relative_tolerance']), 'rel_tol': 0}
    result['display_tolerance_override'] = 0
    if 'bounds' in rule:
        result['numeric_bounds'] = rule['bounds']
    return result

def policy_field(field, policy):
    if 'continuous_index' in policy:
        return _apply_numeric_policy(field, policy)
    result = deepcopy(field)
    renal = policy['renal_continuous']
    if field['kind'] in renal['kinds'] and field['unit'] in renal['units']:
        result['tolerance'] = {'abs_tol': max(renal['absolute_tolerance'], abs(field['expected']) * renal['relative_tolerance']), 'rel_tol': 0}
    return result

VERSION = 'answer_scoring'

BOUNDARY = re.compile('[;\\n]|\\.(?!\\d)')

SCORE = re.compile('\\b(?:(?:total|overall|final)\\s+(?:score\\s*)?(?:of|is|=|:)?|total(?:ing|ling|s)?\\s*(?:of|is|=|:)?|scor(?:e|es|ed)\\s*(?:would\\s+be|is|of|=|:)?)\\s*(?:approximately\\s+|about\\s+)?(?P<n>' + NUM + ')(?!\\w|\\.\\d)', re.I)

def method_spans(text, vocabulary):
    found = []
    for tid, entry in vocabulary.items():
        for label in entry['labels']:
            tokens = re.findall('[a-z0-9]+', canonical(label).lower())
            if not tokens or (len(tokens) == 1 and len(tokens[0]) < 3):
                continue
            pattern = '(?<!\\w)' + '[\\s_()\\-]*'.join(map(re.escape, tokens)) + '(?!\\w)'
            found.extend(((m.start(), m.end(), tid) for m in re.finditer(pattern, text, re.I)))
    longest = {(s, e, t) for s, e, t in found if not any((a <= s and b >= e and (b - a > e - s) for a, b, _ in found))}
    groups = {}
    for s, e, tid in longest:
        groups.setdefault((s, e), set()).add(tid)
    return [{'span': [s, e], 'tool_ids': sorted(ids)} for (s, e), ids in sorted(groups.items())]

def clause(text, start, end):
    left = [m.end() for m in BOUNDARY.finditer(text[:start])]
    right = BOUNDARY.search(text, end)
    lo, hi = (left[-1] if left else 0, right.start() if right else len(text))
    return (lo, hi, text[lo:hi])

def endpoint_words(value):
    """Surface-order equivalence of PUBLIC labels, not clinical inference."""
    value = re.sub('(?<=[a-z])(?=[A-Z])', ' ', value).lower()
    words = re.findall('[a-z]+|\\d+', value)
    numbers = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'seven': '7', 'ten': '10', 'thirty': '30', 'ninety': '90'}
    stop = {'the', 'a', 'an', 'of', 'is', 'are', 'with', 'estimated', 'approximately', 'about', 'risk', 'probability', 'percent', 'percentage', 'in', 'at', 'over', 'within'}
    return {numbers.get(w, w.rstrip('s') if w in ('days', 'years', 'months') else w) for w in words if w not in stop}

def extract(text, vocabulary, catalog=None):
    """No episode, Gold, target answer or reasoning text is accepted."""
    text = canonical(text)
    parsed = _extract_base(text, vocabulary)
    atoms = parsed['atoms']
    spans = method_spans(text, vocabulary)
    for m in SCORE.finditer(text):
        if any((m.start() < a['span'][1] and m.end() > a['span'][0] for a in atoms)):
            continue
        atoms.append({'measure': 'score', 'value': float(m['n']), 'unit': 'points', 'span': list(m.span()), 'text': m.group(), 'lead': lead_label(text, m.start())})
    labels = {}
    for entry in (catalog or {}).values():
        for f in entry.get('fields', []):
            if f['kind'] not in ('integer', 'number') or f['unit'] not in ('dimensionless', 'points', 'count'):
                continue
            key = re.sub('(?<=[a-z])(?=[A-Z])', ' ', f['key']).replace('_', ' ')
            if len(key) < 3 or key.lower() in ('age', 'value', 'total', 'score'):
                continue
            labels.setdefault(key, set()).update(f['aliases'])
    for label, aliases in labels.items():
        pattern = '(?<!\\w)' + '[\\s_-]+'.join(map(re.escape, label.split())) + '\\s*(?:is|of|=|:|≈|~)\\s*(?P<n>' + NUM + ')(?!\\w|\\.\\d)'
        for m in re.finditer(pattern, text, re.I):
            if any((m.start() < a['span'][1] and m.end() > a['span'][0] for a in atoms)):
                continue
            atoms.append({'measure': 'named_scalar', 'value': float(m['n']), 'unit': '', 'field_aliases': sorted(aliases), 'span': list(m.span()), 'text': m.group(), 'lead': normalize(label)})
    for a in atoms:
        start, end = a['span']
        lo, hi, sentence = clause(text, start, end)
        local = [m for m in spans if lo <= m['span'][0] < hi]
        preceding = [m for m in local if m['span'][0] <= start]
        if preceding:
            a['tool_ids'] = preceding[-1]['tool_ids']
        elif len(local) == 1:
            a['tool_ids'] = local[0]['tool_ids']
        else:
            all_ids = {t for m in spans for t in m['tool_ids']}
            a['tool_ids'] = sorted(all_ids) if len(all_ids) == 1 else []
        a['method_ambiguous'] = len(a['tool_ids']) > 1 or (not a['tool_ids'] and len(parsed['method_ids']) > 1)
        before, after = (text[lo:start], text[end:hi])
        a['endpoint_context'] = before.rsplit(',', 1)[-1]
        a['range_endpoint'] = bool(re.search(NUM + '\\s*(?:[-–—]|to|and)\\s*$', before, re.I) or re.match('\\s*(?:[-–—]|to)\\s*' + NUM, after, re.I))
        a['qualified'] = bool(re.search('\\b(?:if|might|may|could|not true|cannot conclude|at least|at most)\\b', before, re.I) or INSUFFICIENT.search(sentence) or re.search('\\b(?:not|rather than|instead of)\\s*$', before, re.I))
        a['explicit_total'] = bool(re.search('\\b(?:total|overall|final|totaling|totalling)\\b', a['text'], re.I) or re.search('\\b(?:total|overall|final)\\s*$', before, re.I))
        a['component'] = bool(a['measure'] == 'score' and (re.match('\\s+(?:for|due to)\\b', after, re.I) and (not re.match('\\s+for\\s+(?:(?:this|the)\\s+)?patient\\b', after, re.I)) or (text[:start].count('(') > text[:start].count(')') and re.match('[-+]?\\d', a['text']) and (not a['explicit_total']) and re.search('[=:]\\s*$', before))))
    insufficiency = parsed['insufficiency_statement'] or bool(re.search('\\b(?:cannot|can not|unable to)\\b.{0,45}\\b(?:evaluat\\w*|establish\\w*)', text, re.I))
    missing_reason = bool(re.search('\\b(?:missing|unavailab\\w*|not provided|not recorded|not measured|insufficient|unknown|not enough|unevaluable)\\b', text, re.I))
    concrete = [a for a in atoms if not (a['qualified'] or a['component'] or a['range_endpoint'])]
    stop = 'INSUFFICIENT_INFORMATION' if insufficiency and missing_reason and (not concrete) else 'CONFLICTING_DECLARATION' if insufficiency and concrete else 'UNSPECIFIED_INABILITY' if insufficiency else 'NO_DECLARATION'
    parsed.update(atoms=sorted(atoms, key=lambda a: a['span']), method_spans=spans, insufficiency_statement=bool(insufficiency), stop_declaration=stop)
    return parsed

def score_branch(parsed, branch):
    required = [f for f in branch['fields'] if f['required']]
    atoms = [a for a in parsed['atoms'] if not a.get('component') and (not a.get('range_endpoint')) and (not a.get('tool_ids') or branch['tool_id'] in a['tool_ids'])]
    checks = []
    for f in required:
        selected = []
        for a in atoms:
            if a['measure'] == 'named_scalar':
                match = bool(set(a['field_aliases']) & set(f['aliases']))
            else:
                match = relevant(a, f, required)
                if a['measure'] == 'probability' and f['unit'] == 'percent':
                    context = a.get('endpoint_context', '')
                    signature = endpoint_words(f['key'])
                    match |= bool(signature and signature <= endpoint_words(context))
            if match:
                item = deepcopy(a)
                if a['measure'] == 'named_scalar':
                    item.update(measure='scalar', unit=f['unit'])
                elif a['measure'] == 'probability':
                    item.update(measure='scalar')
                selected.append(item)
        proxy = deepcopy(parsed)
        proxy['atoms'] = selected
        single = deepcopy(f)
        if any((a['measure'] == 'scalar' for a in selected)):
            single['required'] = True
        check = _score_base_branch(proxy, {**branch, 'fields': [single]})['fields'][0]
        if any((a.get('method_ambiguous') for a in selected)):
            check.update(status='UNRESOLVED', reason='METHOD_AMBIGUOUS')
        checks.append(check)
    states = {c['status'] for c in checks}
    status = next((s for s in ('WRONG', 'UNRESOLVED', 'INCOMPLETE', 'DISPLAY_ONLY') if s in states), 'CORRECT' if checks else 'UNRESOLVED')
    return {'tool_id': branch['tool_id'], 'status': status, 'fields': checks}

def score_parsed(parsed, episode):
    result = {'version': VERSION, 'scope': episode['scope'], 'extracted': parsed, 'branches': [], 'status': 'NOT_APPLICABLE', 'answer_success_strict': None, 'answer_success_display': None, 'information_insufficiency_recognized': None}
    if episode['scope'] == 'INFORMATION_ONLY':
        recognized = parsed['insufficiency_statement']
        result.update(status='INSUFFICIENCY_RECOGNIZED' if recognized else 'INSUFFICIENCY_NOT_RECOGNIZED', information_insufficiency_recognized=recognized)
        return result
    if episode['scope'] != 'FINAL_ANSWER':
        return result
    result.update(answer_success_strict=False, answer_success_display=False)
    if parsed['insufficiency_statement'] and (not parsed['atoms']):
        result['status'] = 'NO_ANSWER'
        return result
    branches = [b for b in episode['branches'] if b['scorable']]
    if parsed['method_ids']:
        branches = [b for b in branches if b['tool_id'] in parsed['method_ids']]
        if not branches:
            result['status'] = 'WRONG_METHOD'
            return result
    result['branches'] = [score_branch(parsed, b) for b in branches]
    statuses = {b['status'] for b in result['branches']}
    result.update(answer_success_strict='CORRECT' in statuses, answer_success_display=bool({'CORRECT', 'DISPLAY_ONLY'} & statuses), status=next((s for s in ('CORRECT', 'DISPLAY_ONLY', 'UNRESOLVED', 'INCOMPLETE', 'WRONG') if s in statuses), 'UNRESOLVED'))
    return result

def score(text, episode, vocabulary, policy, catalog=None):
    parsed = extract(text, vocabulary, catalog)
    original = score_parsed(parsed, episode)
    adjusted = deepcopy(episode)
    for branch in adjusted.get('branches', []):
        branch['fields'] = [policy_field(f, policy) for f in branch['fields']]
    sensitivity = score_parsed(parsed, adjusted)
    original.update(numeric_policy=policy['version'], answer_success_type_aware=sensitivity['answer_success_display'], type_aware_status=sensitivity['status'], type_aware_branches=sensitivity['branches'])
    return original

