"""Pinned local Hugging Face models, independent single-input greedy calls."""
import hashlib
import json

class Model:
    def __init__(self,config,cache_dir,seed):
        import torch
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable in the execution environment')
        torch.set_num_threads(4); set_seed(seed)
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.benchmark=False
        kwargs=dict(revision=config['revision'],cache_dir=cache_dir,local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(config['repo_id'],**kwargs)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model=AutoModelForCausalLM.from_pretrained(config['repo_id'],torch_dtype=dtype,**kwargs).to('cuda:0').eval()
        self.config=config
        self.meta=dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),dtype=str(dtype),repo_id=config['repo_id'],revision=config['revision'],
            loaded_commit=getattr(self.model.config,'_commit_hash',None),
            chat_template_sha256=hashlib.sha256(json.dumps(self.tokenizer.chat_template,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            model_config=self.model.config.to_dict(),batch_size=1,do_sample=False,seed=seed,kv_cache_reuse_across_calls=False)
        if self.meta['loaded_commit'] not in [None,config['revision']]: raise RuntimeError('loaded model commit mismatch')

    def generate(self,messages,max_tokens):
        import torch
        kwargs=dict(tokenize=False,add_generation_prompt=True)
        if self.config['thinking'] is False: kwargs['enable_thinking']=False
        rendered=self.tokenizer.apply_chat_template(messages,**kwargs)
        inputs=self.tokenizer(rendered,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            output=self.model.generate(**inputs,do_sample=False,num_beams=1,repetition_penalty=1.0,
                max_new_tokens=max_tokens,pad_token_id=self.tokenizer.pad_token_id)
        tail=output[0,inputs['input_ids'].shape[1]:]
        return dict(raw=self.tokenizer.decode(tail,skip_special_tokens=True),rendered=rendered,
            n_input_tokens=int(inputs['input_ids'].shape[1]),n_output_tokens=int(tail.shape[0]))
