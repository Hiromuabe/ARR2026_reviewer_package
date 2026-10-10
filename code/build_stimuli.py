"""Create review candidates only; never mark machine drafts as human-reviewed."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from catalog import EVENTS, SURNAMES, GIVEN, ANCHORS

BASE=Path(__file__).resolve().parent

def dumps(x): return json.dumps(x,ensure_ascii=False,sort_keys=True)
def digest(x): return hashlib.sha256(dumps(x).encode()).hexdigest()
def fact(subject,predicate,obj,period,polarity=True):
    return dict(subject=subject,predicate=predicate,object=obj,period=period,polarity=polarity)

def make_candidates():
    result=[]
    for i,(obj,past,base_verb) in enumerate(EVENTS):
        ph='again' if i<40 else 'second'
        subject=GIVEN[i%3]+' '+SURNAMES[i%40]
        decoy=GIVEN[(i%3+1)%3]+' '+SURNAMES[(i+17)%40]
        anchor_person=GIVEN[(i%3+2)%3]+' '+SURNAMES[(i+29)%40]
        t=datetime(2025+i%3,1+i%12,1+i//12,9+i%8,15*(i%4),0)
        t0=t.isoformat(); date=t.strftime('%Y-%m-%d at %H:%M:%S')
        periods=dict(before_2024=[None,'2024-01-01T00:00:00'],year_2024=['2024-01-01T00:00:00','2025-01-01T00:00:00'],after_2024_before_reference=['2025-01-01T00:00:00',t0],before_reference=[None,t0],at_reference=[t0,(t+timedelta(seconds=1)).isoformat()])
        repeat='again' if ph=='again' else 'for the second time in their life'
        event=f'{subject} {past} {repeat}'
        sentences=[f'At the reference time, {event}.',f'The record states that, at the reference time, {event}.',f'At the reference time, it was {subject} who {past} {repeat}.',f'The event recorded at the reference time was that {event}.',f'According to the record, {event} at the reference time.',f'{event[0].upper()+event[1:]} at the reference time, as recorded.']
        anchor=fact(anchor_person,'location',ANCHORS[i%6],'at_reference')
        prior=fact(subject,'event',obj,'before_reference')
        assertion=fact(subject,'event',obj,'at_reference')
        state=dict(assertions=[assertion,anchor],presuppositions=[prior] if ph=='again' else [],current_facts=[assertion,prior,anchor],revision_log=[],presupposition_conflicts=[])
        if ph=='second':
            count_prior=dict(prior,count=1)
            state['assertions'] += [prior,count_prior]
            state['current_facts'].append(dict(count_prior))
        narrow='during the calendar year 2024'
        whole=f'at any time from birth until immediately before {date}'
        def denial(who,span): return f'{who} did not {base_verb} even once {span}.'
        updates=[dict(id='Narrow',text=denial(subject,narrow),fact=fact(subject,'event',obj,'year_2024',False),gold_conflict=False),dict(id='Whole',text=denial(subject,whole),fact=fact(subject,'event',obj,'before_reference',False),gold_conflict=True),dict(id='Unrelated',text=denial(decoy,whole),fact=fact(decoy,'event',obj,'before_reference',False),gold_conflict=False,control_type='person')]
        result.append(dict(id=f'confirm_{i+1:03}',aligned_japanese_id=f'confirm_{i+1:03}',version=2,language='en',phenomenon=ph,subject=subject,reference=t0,periods=periods,utterance=f'The reference time is {date}. '+sentences[i%6]+f' At the same time, {anchor_person} is in the {anchor["object"]}.',target_description=base_verb,target_past_information=prior,target_role='presuppositions' if ph=='again' else 'assertions',anchor=anchor,gold_state=state,updates=updates,object_aliases={obj:[obj],anchor['object']:[anchor['object']]},review_status='candidate_not_human_validated',narrow_span=narrow,whole_span=whole))
    return result


def novelty(candidates):
    old=json.loads((BASE/'exploration_reference.json').read_text())
    objects=set(); sentences=set()
    for x in old:
        objects.add(x['gold_state_A']['assertions'][0]['object']); objects.update(x.get('object_aliases',[]))
        for k in ['sentence_A','sentence_B']: sentences.add(x[k])
    target_overlap=[x['id'] for x in candidates if x['target_past_information']['object'] in objects]
    source_overlap=[x['id'] for x in candidates if x['utterance'] in sentences]
    assert not target_overlap and not source_overlap
    return dict(old_base_stimuli=len(old),candidate_base_stimuli=len(candidates),target_exact_overlap=target_overlap,
        utterance_exact_overlap=source_overlap,unique_targets=len({x['target_past_information']['object'] for x in candidates}),
        limitation='Lexical exact-match check; conceptual novelty also requires human confirmation. Candidate drafts authored by assistant, not target models.')


def export(candidates,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'stimuli_candidates.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n')
    (out/'novelty_audit.json').write_text(json.dumps(novelty(candidates),ensure_ascii=False,indent=2)+'\n')
    fields=['id','stimulus_sha256','reference','phenomenon','utterance','Narrow','Whole','Unrelated',
        'past_information_paraphrase','current_assertion_paraphrase','target_past_information_valid',
        'Narrow_conflict','Whole_conflict','Unrelated_conflict','initial_state_valid','conceptually_new','decision','notes']
    for reviewer in ['reviewer_1','reviewer_2']:
        folder=out/reviewer; folder.mkdir()
        with (folder/'annotations.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader()
            for x in candidates:
                row={k:'' for k in fields}; row.update(id=x['id'],stimulus_sha256=digest(x),reference=x['reference'],phenomenon=x['phenomenon'],utterance=x['utterance'])
                row.update({u['id']:u['text'] for u in x['updates']}); writer.writerow(row)
        with (folder/'states_second_pass.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=['id','stimulus_sha256','initial_state','periods']); writer.writeheader()
            for x in candidates: writer.writerow(dict(id=x['id'],stimulus_sha256=digest(x),initial_state=dumps(x['gold_state']),periods=dumps(x['periods'])))
        (folder/'reviewer.json').write_text(json.dumps(dict(reviewer_id='',is_human=False,independent=False,reviewed_at='',attestation=''),ensure_ascii=False,indent=2)+'\n')
    (out/'original_exploration_targets.json').write_text(json.dumps(sorted({x['gold_state_A']['assertions'][0]['object'] for x in json.loads((BASE/'exploration_reference.json').read_text())}),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=BASE/'review_packet')
    export(make_candidates(),parser.parse_args().output)
