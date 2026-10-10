#!/usr/bin/env python3
"""Portable, standard-library-only reanalysis of the actual paper calls."""
import argparse, collections, csv, hashlib, json, random, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'code'
sys.path.insert(0,str(F))
import semantics as s, prompts as p
PARA_RULE='Each positive fact says that at least one occurrence lies within its named interval; each negative fact excludes all occurrences from its interval. For matching subject, predicate and object, these statements conflict only when the negative interval contains the entire positive interval. Partial exclusion leaves existence outside that subinterval possible. Do not infer additional occurrences. Exact counts remain exact counts.'
PARA_UPDATE='Revise the supplied state in response to the new information. Remove a prior assertion or presupposition only if they cannot both be true under the interval rules. Remove it from its role array and current_facts. Record each removal in revision_log using its original role; for a removed presupposition also record the pair in presupposition_conflicts. Keep every compatible fact and its role, time interval, polarity and exact count. Store the incoming proposition in current_facts. Keep all existing history entries. Use only the provided state; do not reconstruct an earlier utterance or create missing facts.'
def load(path):return json.loads((ROOT/path).read_text())
def jsonl(path):return [json.loads(l) for l in (ROOT/path).read_text().splitlines()]
def bootstrap(values,replication=False):
 rng=random.Random(42 if replication else 271828);n=len(values)
 draws=sorted(sum(rng.choices(values,k=n) if replication else [rng.choice(values) for _ in values])/n for _ in range(10000))
 if replication:return [100*draws[249],100*draws[9749]]
 def q(prob):
  x=(len(draws)-1)*prob;i=int(x);return draws[i]+(draws[min(i+1,len(draws)-1)]-draws[i])*(x-i)
 return [100*q(.025),100*q(.975)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
 manifest=load('manifest.json')
 for name,h in manifest['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,('checksum',name)
 base=load('data/basic_problems.json');lookup={st['id']:st for st in base}
 assert len(base)==40 and len(lookup)==40
 variants={(v['family'],v['stimulus']['id']):v['stimulus'] for v in load('data/variants.json')}
 # Rebuild the original synthetic candidates and check actual inputs against them.
 from build_stimuli import make_candidates
 built={st['id']:st for st in make_candidates()}
 for st in base:
  for key in ['gold_state','periods','utterance','updates']:assert st[key]==built[st['id']][key],('generation',st['id'],key)
 for case in load('data/challenge_cases.json'):
  st=lookup[case['id']];u=next(u for u in st['updates'] if u['id']==case['update']);target=st['target_past_information'];fact=u['fact']
  a,b=st['periods'][fact['period']];c,d=st['periods'][target['period']]
  identity=all(target[k]==fact[k] for k in ['subject','predicate','object'])
  expected=identity and (a is None or (c is not None and a<=c)) and b>=d
  assert expected is u['gold_conflict']
  assert case['gold_state_after_update']==s.oracle(st['gold_state'],fact,st)
 records=jsonl('data/records.jsonl');prompts={r['record_id']:r for r in jsonl('data/prompts.jsonl')};source_cache={};groups=collections.defaultdict(list)
 judges={(w['experiment'],w['record']['id'],w['record'].get('update','Narrow')):w['record']['decision'] for w in records if w['record'].get('stage')=='judge'}
 configs=load('configs/experiments.json')
 assert len(prompts)==len(records)==2720
 for w in records:
  g=w['experiment'];r=w['record'];stage=r.get('stage','update');uid=w['record_id'];pr=prompts[uid];cfg=configs[g]
  st=variants[(g.removeprefix('intervention_'),r['id'])] if g.startswith('intervention_') else lookup[r['id']]
  u=next(u for u in st['updates'] if u['id']==r.get('update','Narrow'))
  assert pr['messages']==r['messages'] and pr['rendered']==r['rendered'] and pr['settings']==cfg
  if stage=='judge':text=p.judge(st,u)
  else:
   arm=r['condition'];match=re.search(r'Auxiliary judgment: \{"conflict": (true|false)\}',r['messages'][1]['content']);cue=(match.group(1)=='true') if match else None
   expected=None if arm=='NoCue' else (u['gold_conflict'] if arm=='Gold' else (not u['gold_conflict'] if arm=='Flipped' else judges[(g,r['id'],u['id'])]))
   assert cue is expected,('cue',uid)
   text=p.updater(st,u,cue)
   if g=='intervention_paraphrase_instructions':text=text.replace(p.RULE,PARA_RULE).replace(p.UPDATE,PARA_UPDATE)
  assert r['messages']==p.messages(text),('prompt',uid)
  if cfg['do_sample']:
   expected_seed=42+int(r['id'].split('_')[-1])*100
   if not g.startswith('intervention_'):expected_seed+=['Narrow','Whole','Unrelated'].index(u['id'])*10+{'NoCue':0,'Self':1,'Gold':2,None:3}[r.get('condition')]
   assert r['seed']==expected_seed
  if stage=='judge':
   value,err=p.parse_judge(r.get('final_answer',r.get('raw')));assert value==r['decision'];r['_correct']=value is u['gold_conflict']
  else:
   state,err=s.parse_state(r.get('final_answer',r.get('raw')),st) if r.get('error') not in ['thinking_not_completed','invalid_judge'] else (None,r['error'])
   ev=s.score(state,st,u);assert state==r['parsed'] and ev==r['evaluation'],('score',uid)
   r['_evaluation']=ev
   if state is not None and u['id']=='Narrow':
    target=st['target_past_information'];lo,hi=st['periods'][target['period']];retained=False
    for fact in state['presuppositions']:
     obj=next((canonical for canonical,aliases in st['object_aliases'].items() if fact['object'] in aliases),fact['object'])
     if (fact['subject'],fact['predicate'],obj,fact['polarity'])==(target['subject'],target['predicate'],target['object'],True):
      a,b=st['periods'][fact['period']];retained|=(lo is None or (a is not None and lo<=a)) and b<=hi
    assert int(not retained)==ev['target_information_loss'],('independent loss',uid)
  groups[g].append(r)
 judgment=[];arms=[];paired=[]
 for g,rows in groups.items():
  for u in ['Narrow','Whole','Unrelated']:
   js=[r for r in rows if r.get('stage')=='judge' and r['update']==u]
   if js:judgment.append(dict(experiment=g,update=u,n=len(js),correct=sum(r['_correct'] for r in js)))
  ix={}
  for arm in ['NoCue','Self','Gold','Flipped']:
   us=[r for r in rows if r.get('stage','update')=='update' and r.get('update','Narrow')=='Narrow' and r['condition']==arm];vs=[r for r in us if r['_evaluation'] is not None]
   if us:
    assert len(us)==40,(g,arm,len(us))
    arms.append(dict(experiment=g,condition=arm,n=40,valid=len(vs),invalid=40-len(vs),loss=sum(r['_evaluation']['target_information_loss'] for r in vs),full_correct=sum(r['_evaluation']['update_correct'] for r in vs)))
    ix.update({(r['id'],arm):r for r in us})
  a='Flipped' if g.startswith('intervention_') else 'Self';b='Gold'
  if not (g.startswith('intervention_') or g.endswith('_greedy')):continue
  ids=sorted({id_ for id_,arm in ix});ids=[id_ for id_ in ids if (id_,a) in ix and (id_,b) in ix and ix[(id_,a)]['_evaluation'] is not None and ix[(id_,b)]['_evaluation'] is not None]
  # Replication's historical bootstrap uses order of first appearance, not sorted IDs.
  if g.startswith('intervention_'):ids=list(dict.fromkeys(r['id'] for r in rows if r['id'] in ids))
  diffs=[ix[(id_,a)]['_evaluation']['target_information_loss']-ix[(id_,b)]['_evaluation']['target_information_loss'] for id_ in ids]
  item=dict(experiment=g,comparison=a+' minus '+b,n=len(ids),a_loss=sum(ix[(id_,a)]['_evaluation']['target_information_loss'] for id_ in ids),b_loss=sum(ix[(id_,b)]['_evaluation']['target_information_loss'] for id_ in ids),difference_pp=100*sum(diffs)/len(diffs),ci95_pp=bootstrap(diffs,g.startswith('intervention_')))
  if g.startswith('intervention_'):
   aa=next(r for r in arms if r['experiment']==g and r['condition']==a);bb=next(r for r in arms if r['experiment']==g and r['condition']==b)
   item['all40_missing_bounds_pp']=[100*(aa['loss']-bb['loss']-bb['invalid'])/40,100*(aa['loss']+aa['invalid']-bb['loss'])/40]
   for id_ in {id_ for id_,arm in ix}:
    ra,rb=ix[(id_,a)],ix[(id_,b)];assert ra['seed']==rb['seed']
    normalize=lambda t:re.sub(r'Auxiliary judgment: \{"conflict": (true|false)\}','Auxiliary judgment: BOOL',t)
    assert normalize(ra['rendered'])==normalize(rb['rendered'])
  else:
   for id_ in {id_ for id_,arm in ix}:
    assert ix[(id_,'Self')]['messages']==ix[(id_,'Flipped')]['messages']
  paired.append(item)
 rs=[r for r in groups['qwen32_sampled_on'] if r.get('condition')=='Gold' and r.get('update')=='Narrow']
 ev=[r['_evaluation'] for r in rs];wrong=[v for v in ev if v is not None and not v['update_correct']]
 appendix=dict(n=40,full_correct=sum(v is not None and v['update_correct'] for v in ev),invalid=sum(v is None for v in ev),presupposition_loss=sum(v is not None and v['target_information_loss'] for v in ev),retained_other_failure=sum(v is not None and not v['target_information_loss'] and not v['update_correct'] for v in ev),requirement_failures={k:sum((not v[k]) if k=='unrelated_retained' else bool(v[k]) for v in wrong) for k in ['unrelated_retained','unsupported_role_facts','unsupported_revisions','unsupported_conflicts']})
 result=dict(verified_records=len(records),verified_prompts=len(prompts),judgment=judgment,arms=arms,paired=paired,appendix=appendix)
 # Compare every reported table and appendix component to the reviewed manuscript values.
 expected=load('configs/expected_results.json')
 for section in ['judgment','arms','paired']:
  actual={tuple(row[k] for k in (['experiment','update'] if section=='judgment' else ['experiment','condition'] if section=='arms' else ['experiment'])):row for row in result[section]}
  for row in expected[section]:
   key=tuple(row[k] for k in (['experiment','update'] if section=='judgment' else ['experiment','condition'] if section=='arms' else ['experiment']))
   for k,v in row.items():
    av=actual[key][k]
    assert av==v or (isinstance(v,float) and abs(av-v)<1e-8) or (isinstance(v,list) and len(av)==len(v) and all(abs(x-y)<1e-8 for x,y in zip(av,v))),('paper mismatch',section,key,k,av,v)
 assert appendix==expected['appendix'],('appendix',appendix,expected['appendix'])
 args.output_dir.mkdir(parents=True,exist_ok=True)
 (args.output_dir/'recomputed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 for section in ['judgment','arms','paired']:
  rows=result[section];keys=list(dict.fromkeys(k for row in rows for k in row))
  with (args.output_dir/(section+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 print('PASS: checksums, 40 generated stimuli, 120 formal labels, 2720 raw records/prompts, scoring, reported results and appendix counts.');print(args.output_dir)
if __name__=='__main__':main()
