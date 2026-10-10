"""Case-specific finite interval ontology and role-sensitive semantic scoring."""
import copy
import json

FIELDS=('assertions','presuppositions','current_facts','revision_log','presupposition_conflicts')
FACT_KEYS={'subject','predicate','object','period','polarity'}


def canonical(f,stim):
    out=dict(f)
    for obj,aliases in stim['object_aliases'].items():
        if out['object'] in aliases: out['object']=obj; break
    return out


def valid_fact(f,stim):
    return isinstance(f,dict) and (set(f)==FACT_KEYS or (set(f)==FACT_KEYS|{'count'} and type(f['count']) is int and f['count']>0 and f['polarity'] is True and f['predicate']=='event')) and all(isinstance(f[k],str) and f[k] for k in ['subject','object']) and f['predicate'] in ['activity','event','location'] and f['period'] in stim['periods'] and type(f['polarity']) is bool


def parse_state(raw,stim):
    try:
        state=json.loads(raw)
        if not isinstance(state,dict) or set(state)!=set(FIELDS) or any(not isinstance(state[k],list) for k in FIELDS): raise ValueError('exactly five list fields required')
        if any(not valid_fact(f,stim) for k in FIELDS[:3] for f in state[k]): raise ValueError('invalid fact')
        for r in state['revision_log']:
            if not isinstance(r,dict) or set(r)!={'type','fact'} or r['type'] not in ['retract_assertion','retract_presupposition'] or not valid_fact(r['fact'],stim): raise ValueError('invalid revision')
        for r in state['presupposition_conflicts']:
            if not isinstance(r,dict) or set(r)!={'prior','incoming'} or not valid_fact(r['prior'],stim) or not valid_fact(r['incoming'],stim): raise ValueError('invalid conflict')
        return state,None
    except (ValueError,TypeError,KeyError,AttributeError) as exc: return None,str(exc)


def covers(outer,inner,stim):
    a,b=stim['periods'][outer]; c,d=stim['periods'][inner]
    return (a is None or (c is not None and a<=c)) and b>=d


def identity(f,stim):
    f=canonical(f,stim)
    return tuple(f[k] for k in ['subject','predicate','object'])


def conflicts(a,b,stim):
    if identity(a,stim)!=identity(b,stim) or a['polarity']==b['polarity']: return False
    positive,negative=(a,b) if a['polarity'] else (b,a)
    return covers(negative['period'],positive['period'],stim)


def entails(facts,target,stim):
    candidates=[f for f in facts if identity(f,stim)==identity(target,stim) and f['polarity']==target['polarity']]
    if 'count' in target:
        return any(f.get('count')==target['count'] and stim['periods'][f['period']]==stim['periods'][target['period']] for f in candidates)
    if target['polarity']:
        return any(covers(target['period'],f['period'],stim) for f in candidates)
    # Union of adjacent negative intervals may entail a broader negative.
    start,end=stim['periods'][target['period']]; cursor=start or ''
    for a,b in sorted((a or '',b) for a,b in (stim['periods'][f['period']] for f in candidates)):
        if a>cursor: break
        if b>cursor: cursor=b
        if cursor>=end: return True
    return False


def equal_fact(a,b,stim): return canonical(a,stim)==canonical(b,stim)


def oracle(initial,incoming,stim):
    out=copy.deepcopy(initial); removed=[]
    for field,kind in [('assertions','retract_assertion'),('presuppositions','retract_presupposition')]:
        out[field]=[]
        for fact in initial[field]:
            if conflicts(fact,incoming,stim):
                removed.append(fact); out['revision_log'].append(dict(type=kind,fact=fact))
                if field=='presuppositions': out['presupposition_conflicts'].append(dict(prior=fact,incoming=incoming))
            else: out[field].append(fact)
    out['current_facts']=[f for f in initial['current_facts'] if not any(equal_fact(f,r,stim) for r in removed)]
    if not any(equal_fact(f,incoming,stim) for f in out['current_facts']): out['current_facts'].append(copy.deepcopy(incoming))
    return out


def active_facts(state): return [f for field in FIELDS[:3] for f in state[field]]


def answer_label(state,query,stim):
    facts=active_facts(state)
    opposite=dict(query,polarity=not query['polarity'])
    yes=entails(facts,query,stim); no=entails(facts,opposite,stim)
    return 'inconsistent' if yes and no else 'yes' if yes else 'no' if no else 'unknown'


def score(updated,stim,update):
    if updated is None: return None
    initial=stim['gold_state']; incoming=update['fact']; expected=oracle(initial,incoming,stim)
    prior=stim['target_past_information']; target_role=stim.get('target_role','presuppositions'); retained=entails(updated[target_role],prior,stim)
    nonconflict=not update['gold_conflict']
    losses=[dict(field=field,fact=f) for field in FIELDS[:3] for f in expected[field]
        if any(equal_fact(f,old,stim) for old in initial[field]) and not entails(updated[field],f,stim)]
    additions=[dict(field=field,fact=f) for field in FIELDS[:3] for f in updated[field] if not entails(expected[field],f,stim)]
    # Updates never conflict with current assertions or the background anchor.
    unaffected=[dict(field=field,fact=f) for field in FIELDS[:3] for f in initial[field]
        if not conflicts(f,incoming,stim)]
    unrelated_retained=all(entails(updated[x['field']],x['fact'],stim) for x in unaffected)
    background_retained=entails(updated['assertions'],stim['anchor'],stim) and entails(updated['current_facts'],stim['anchor'],stim)
    incoming_stored=entails(updated['current_facts'],incoming,stim)
    revoked_correctly=not retained and not entails(updated['current_facts'],prior,stim)
    valid_revisions=[]; bad_revisions=[]
    for rev in updated['revision_log']:
        field='presuppositions' if rev['type']=='retract_presupposition' else 'assertions'
        ok=any(equal_fact(rev['fact'],f,stim) for f in initial[field]) and conflicts(rev['fact'],incoming,stim)
        (valid_revisions if ok else bad_revisions).append(rev)
    valid_pairs=[]; bad_pairs=[]
    for pair in updated['presupposition_conflicts']:
        ok=any(equal_fact(pair['prior'],f,stim) for f in initial['presuppositions']) and entails([incoming],pair['incoming'],stim) and conflicts(pair['prior'],pair['incoming'],stim)
        (valid_pairs if ok else bad_pairs).append(pair)
    expected_rev=expected['revision_log']
    history_complete=all(any(r['type']==x['type'] and equal_fact(r['fact'],x['fact'],stim) for r in valid_revisions) for x in expected_rev) and all(any(equal_fact(pair['prior'],x['prior'],stim) and equal_fact(pair['incoming'],x['incoming'],stim) for pair in valid_pairs) for x in expected['presupposition_conflicts'])
    action_ok=retained if nonconflict else revoked_correctly
    four=action_ok and incoming_stored and unrelated_retained
    return dict(presupposition_loss=int(not retained) if nonconflict else None,
        target_information_loss=int(not retained) if nonconflict else None,target_role=target_role,
        presupposition_retained=retained,legitimate_retraction=revoked_correctly if not nonconflict else None,
        incoming_stored=incoming_stored,unrelated_retained=unrelated_retained,background_retained=background_retained,
        all_four_information_goals=bool(four),update_correct=bool(four and not additions and not bad_revisions and not bad_pairs and history_complete),
        actual_losses=losses,unsupported_role_facts=additions,unsupported_revisions=bad_revisions,unsupported_conflicts=bad_pairs,
        recorded_conflict=bool(updated['presupposition_conflicts']),justified_conflict_detected=bool(valid_pairs or valid_revisions),
        target_retained_in_active_facts=entails(active_facts(updated),prior,stim),
        target_retained_in_current_facts=entails(updated['current_facts'],prior,stim))
