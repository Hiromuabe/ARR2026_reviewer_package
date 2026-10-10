#!/usr/bin/env python3
"""Replay exactly the saved inputs. Self cues are historical, not newly generated."""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'code'
sys.path.insert(0,str(B))
from generation import split_output
import semantics as s, prompts as p

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--experiment',required=True);ap.add_argument('--cache-dir');ap.add_argument('--output-dir',type=Path);ap.add_argument('--limit',type=int);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--fresh-self',action='store_true',help='Use newly generated judgments for Self; input then differs from historical saved messages.');args=ap.parse_args()
 configs=json.loads((ROOT/'configs/experiments.json').read_text());assert args.experiment in configs,list(configs)
 cfg=configs[args.experiment];rows=[json.loads(l) for l in (ROOT/'data/prompts.jsonl').read_text().splitlines() if json.loads(l)['experiment']==args.experiment]
 if args.fresh_self:
  # NoCue has no conversational dependence. Then judge before dependent Self.
  rows.sort(key=lambda r:0 if r['condition']=='NoCue' else 1 if r['stage']=='judge' else 2)
 if args.limit is not None:
  if args.limit<=0:ap.error('--limit must be positive')
  rows=rows[:args.limit]
 if args.dry_run:print(json.dumps(dict(experiment=args.experiment,calls=len(rows),settings=cfg,fresh_self=args.fresh_self),indent=2));return
 if not args.cache_dir or not args.output_dir:ap.error('--cache-dir and --output-dir required unless --dry-run')
 if args.output_dir.resolve().is_relative_to(ROOT):ap.error('Place new inference outputs outside the immutable reviewer directory')
 args.output_dir.mkdir(parents=True,exist_ok=False)
 from backend import Model
 from transformers import set_seed
 import torch
 model=Model(cfg['model'],args.cache_dir,42)
 if cfg['enable_thinking']:
  end=model.tokenizer.convert_tokens_to_ids('</think>');assert model.tokenizer.convert_ids_to_tokens(end)=='</think>'
 else:end=None
 base={st['id']:st for st in json.loads((ROOT/'data/basic_problems.json').read_text())}
 variants={(v['family'],v['stimulus']['id']):v['stimulus'] for v in json.loads((ROOT/'data/variants.json').read_text())}
 (args.output_dir/'metadata.json').write_text(json.dumps(dict(settings=cfg,runtime=model.meta,fresh_self=args.fresh_self,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 decisions={}
 for index,row in enumerate(rows):
  st=variants[(args.experiment.removeprefix('intervention_'),row['id'])] if args.experiment.startswith('intervention_') else base[row['id']]
  u=next(u for u in st['updates'] if u['id']==row['update']);msgs=row['messages']
  if args.fresh_self and row['condition']=='Self':
   decision=decisions.get((row['id'],row['update']))
   if decision is None:
    rec=dict(record_id=row['record_id'],id=row['id'],stage='update',update=row['update'],condition='Self',error='invalid_judge',parsed=None,evaluation=None,skipped=True)
    with (args.output_dir/'results.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    continue
   msgs=p.messages(p.updater(st,u,decision))
  kwargs=dict(tokenize=False,add_generation_prompt=True)
  if cfg['enable_thinking'] is not None:kwargs['enable_thinking']=cfg['enable_thinking']
  rendered=model.tokenizer.apply_chat_template(msgs,**kwargs)
  if not (args.fresh_self and row['condition']=='Self'):assert rendered==row['rendered'],'Pinned tokenizer rendered a different prompt'
  set_seed(row['seed']);inputs=model.tokenizer(rendered,return_tensors='pt').to('cuda:0')
  options=dict(do_sample=cfg['do_sample'],num_beams=1,repetition_penalty=1.0,max_new_tokens=row['max_new_tokens'],pad_token_id=model.tokenizer.pad_token_id)
  if cfg['do_sample']:options.update(temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'],min_p=cfg['min_p'])
  start=time.monotonic()
  with torch.inference_mode():generated=model.model.generate(**inputs,**options)
  ids=generated[0,inputs['input_ids'].shape[1]:].tolist();answer_ids,thinking,error=split_output(ids,bool(cfg['enable_thinking']),end)
  answer=model.tokenizer.decode(answer_ids,skip_special_tokens=True).strip()
  rec=dict(record_id=row['record_id'],id=row['id'],stage=row['stage'],update=row['update'],condition=row['condition'],seed=row['seed'],messages=msgs,rendered=rendered,raw=model.tokenizer.decode(ids,skip_special_tokens=True),final_answer=answer,n_output_tokens=len(ids),n_thinking_tokens=len(thinking),seconds=time.monotonic()-start,hit_token_limit=len(ids)>=row['max_new_tokens'])
  if row['stage']=='judge':
   decision,err=p.parse_judge(answer) if error is None else (None,error);decisions[(row['id'],row['update'])]=decision;rec.update(decision=decision,error=err,correct=decision is u['gold_conflict'])
  else:
   state,err=s.parse_state(answer,st) if error is None else (None,error);rec.update(parsed=state,error=err,evaluation=s.score(state,st,u))
  with (args.output_dir/'results.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
  print(index+1,len(rows),row['record_id'],rec['error'],flush=True)
if __name__=='__main__':main()
