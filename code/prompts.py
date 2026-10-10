"""English prompts; cue provenance is hidden; original stimulus is not re-presented."""
import json
SYSTEM='This is an experiment on semantic judgments and memory updates. Use only the input. When JSON is requested, return only a raw JSON object, with no code fences or explanation. For a proposition question, return only yes, no, unknown, or inconsistent.'
RULE='''A positive proposition without count means that the event or condition held at least once somewhere in its interval. A negative proposition means it never held anywhere in that interval. event denotes an event; location denotes a location condition.
For the same person, predicate, and object, a negative interval covering the entire positive interval is incompatible with that positive proposition. Denying only part of a broad past interval does not rule out an event elsewhere in that interval. Different persons or objects do not conflict. Compare intervals, not merely interval names.
Do not add unstated events or assumptions from background knowledge.'''
SCHEMA='''The state contains exactly five arrays: assertions, presuppositions, current_facts, revision_log, presupposition_conflicts.
A proposition has {"subject":"person","predicate":"event","object":"event description","period":"interval name","polarity":true}. predicate is event, location, or activity. polarity true is positive; false is negative.
A positive event proposition may additionally have count, a positive integer giving its exact number of occurrences in that interval. count:1 means exactly one; a positive proposition without count specifies existence without a number. Preserve count information.
assertions contains asserted information. presuppositions contains active presupposed information. current_facts contains all currently accepted propositions, including assertions and presuppositions.
revision_log entries are {"type":"retract_assertion","fact":old assertion} or {"type":"retract_presupposition","fact":old presupposition}. presupposition_conflicts entries are {"prior":retracted presupposition,"incoming":conflicting new proposition}. Distinguish assertion revision from presupposition conflict.'''
UPDATE='''Update using only the input state and the new information. Do not reconstruct the original utterance to supply presuppositions.
Retract only existing assertions or presuppositions incompatible with the new information, removing them from their original arrays and current_facts. Log retracted assertions in revision_log. Log retracted presuppositions in both revision_log and presupposition_conflicts.
Preserve the content, role, interval, polarity, and count of all other information. Add the new information to current_facts, not to presuppositions. Do not invent retractions of information absent from the input. Retain existing histories.'''
ADVISORY='Consider the state and new information, and use the auxiliary judgment as evidence when deciding the update.'
def dumps(x): return json.dumps(x,ensure_ascii=False,sort_keys=True)
def messages(text): return [dict(role='system',content=SYSTEM),dict(role='user',content=text)]
def time_text(stim):
    return ('The reference timestamp is '+stim['reference']+' (Asia/Tokyo). The exact date of birth is unspecified. Intervals include their start and exclude their end. at_reference is a one-second observation interval at the reference timestamp.\nNamed intervals: '+dumps(stim['periods']))
def common(stim): return time_text(stim)+'\n'+RULE+'\n'+SCHEMA
def judge(stim,update):
    return common(stim)+'\nState: '+dumps(stim['gold_state'])+'\nNew information: '+update['text']+'\nDoes the new information conflict with the existing past-event information in the state? Return only {"conflict":true} or {"conflict":false}.'
def updater(stim,update,decision=None):
    text=common(stim)+'\n'+UPDATE+('\n'+ADVISORY if decision is not None else '')+'\nState: '+dumps(stim['gold_state'])+'\nNew information: '+update['text']
    if decision is not None: text+='\nAuxiliary judgment: '+dumps({'conflict':decision})
    return text+'\nReturn only the updated state as raw JSON.'
def queries(stim,update):
    flip=int(stim['id'].rsplit('_',1)[1])%2==0
    base=[('past',stim['target_past_information']),('incoming',update['fact']),('background',stim['anchor'])]
    return [(kind,dict(fact,polarity=not fact['polarity']) if flip else dict(fact)) for kind,fact in base]
def readout(stim,state_text,query):
    return (common(stim)+'\nEvaluate the queried proposition using only the updated state. The original utterance and new-information text are not provided. Read revision histories, distinguishing recorded information from currently accepted information.\nUpdated state: '+state_text+'\nQueried proposition: '+dumps(query)+'\nAnswer yes if currently accepted information supports it, no if it supports its opposite, unknown if neither, and inconsistent if both. Return only one of those four words.')
def parse_judge(raw):
    try:
        obj=json.loads(raw)
        if not isinstance(obj,dict) or set(obj)!={'conflict'} or type(obj['conflict']) is not bool: raise ValueError('exactly one boolean field required')
        return obj['conflict'],None
    except (ValueError,TypeError) as exc: return None,str(exc)
def parse_readout(raw):
    value=raw.strip()
    return (value,None) if value in ['yes','no','unknown','inconsistent'] else (None,'readout_format_error')
