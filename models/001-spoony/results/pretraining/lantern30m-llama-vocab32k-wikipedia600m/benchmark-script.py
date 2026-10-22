import json,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from configs.model.model import build_model
from configs.data.dataloader import chunking_config,train_loader_config
from configs.training.optimizer import optimizer_config
from configs.training.precision import precision_config
from configs.training.scheduler import scheduler_config
from src.data.tokenizer import ByteLevelBPETokenizer
from src.data.serialization import TokenStore
from src.data.dataloader import build_split_dataloader
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.precision import build_precision
from src.training.step import train_step

torch.set_num_threads(4);torch.manual_seed(42)
root=Path('data/benchmark/vocab32k/train');tokpath=Path('configs/tokenizers/wikipedia-en-32768.json')
if not root.exists():
 encoder=Tokenizer.from_file(str(tokpath));docs=[];total=0
 for i,line in enumerate(Path('data/raw/tokenizer32k/train.jsonl').open()):
  ids=encoder.encode(json.loads(line)['text']).ids+[1];ids=ids[:1200000-total];docs.append((str(i),ids));total+=len(ids)
  if total==1200000:break
 TokenStore(root).write_documents(docs,tokenizer_sha256='benchmark-only')
tok=ByteLevelBPETokenizer.load(tokpath);model,_=build_model(tok);model.cuda()
opt=build_optimizer(model,optimizer_config);scheduler=build_scheduler(opt,scheduler_config);ctx,scaler=build_precision(torch.device('cuda'),precision_config)
loader=build_split_dataloader(root,chunking_config,train_loader_config);it=iter(loader)
def update():
 batches=[next(it) for _ in range(4)]
 return train_step(model,batches,opt,scheduler,ctx,scaler,torch.device('cuda'),1.)
for _ in range(5):update()
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();rows=[]
for _ in range(3):
 start=time.perf_counter()
 for _ in range(20):loss,ok=update();assert ok
 torch.cuda.synchronize();seconds=time.perf_counter()-start;tokens=20*2*2048*4
 row={'updates':20,'tokens':tokens,'seconds':seconds,'seconds_per_million':seconds*1e6/tokens};rows.append(row);print(json.dumps(row),flush=True)
s=sum(r['seconds'] for r in rows);n=sum(r['tokens'] for r in rows)
r={'measurements':rows,'seconds_per_million_tokens':s*1e6/n,'tokens_per_second':n/s,'training_hours_600m':s*600e6/n/3600,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'parameters':sum(p.numel() for p in model.parameters()),'vocabulary':tok.vocab_size,'microbatch':2,'context':2048,'accumulation':4,'precision':'BF16','source':'actual Wikipedia text tokenized with the new 32768-token tokenizer','excludes':'validation, checkpoint writes and corpus preparation'}
out=Path('results/pretraining/lantern30m-llama-vocab32k-wikipedia600m');out.mkdir(parents=True,exist_ok=True);(out/'benchmark.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
