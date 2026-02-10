import enum
import os
import json
from tqdm import tqdm
import argparse

from utils import call_qwen,call_gpt,call_llm_openrouter_api
from funcs_eval import extract_gold_context,single_exact_match,single_f1_score
import random

QA_MAX_TRY=3
from funcs_eval import HOTPOTQA_PREFIX,HOTPOTQA_CORPUS_PATH,HOTPOTQA_QA_PATH,MUSIQUE_CORPUS_PATH,MUSIQUE_QA_PATH

def llm_run_qa(bench_name,model_name,context_type,max_qa_items):
    
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    all_corpus="\n\n".join(docs)

    
    data=json.load(open(EVAL_DATA_PATH,"r"))[:max_qa_items]
    
    name_base=f"vanilla_{model_name}"
    
    if context_type=="no_context":
        name_base+="_no_ctx"
    elif context_type=="gold_context":
        name_base+="_gold_ctx"
    elif context_type=="random_ten_psg":
        name_base+="_random_ten_psg"
    elif context_type=="ten_psg":
        name_base+="_ten_psg"
    else:
        raise ValueError(f"Invalid context_type: {context_type}")
    
    res_path=f"{RES_DIR}/pred_{name_base}.json"
    metric_path = f"{RES_DIR}/metric_{name_base}.json"
    
    pred_list=[] 
    total_em=0
    total_f1=0
    n=0
    for idx,item in tqdm(enumerate(data)):
        id=item.get('_id', item.get('id', None))
        q=item['question']
        gold=item['answer']
        
        prefix=HOTPOTQA_PREFIX
        if context_type=="no_context":
            prompt=prefix+f"No information is retrieved. Answer the question based on your knowledge.\n\nQuestion: {q}. "
        
        elif context_type=="gold_context":
            gold_ctx=extract_gold_context(item,bench_name)
            prompt=prefix+f"\n\nContext:{gold_ctx}.\n\nQuestion: {q}. "
                
        elif context_type=="ten_psg":
            if bench_name!="hotpotqa":
                raise ValueError(f"ten_psg context_type not supported for {bench_name}: {context_type}")
            ten_psg=""
            for x in item["context"]:
                ten_psg+="\nCaption:"+x[0]+"\nContent:\n"+"".join(x[1])
            prompt=prefix+f"\n\nContext:{ten_psg}.\n\nQuestion: {q}. "
        else:
            raise ValueError(f"Invalid context_type: {context_type}")
            
        try_counts=0
        while True:
            if try_counts>=QA_MAX_TRY:
                break
            if model_name=="Qwen2.5-32B-Instruct":
                pred=call_qwen(prompt)
            elif model_name=="gpt-4o":
                pred=call_gpt(prompt)
            else: 
                pred=call_llm_openrouter_api(prompt)
                
            try_counts+=1
            
            if len(pred)>0:
                break
            
            print(f"trying again, {try_counts}-try (max:{QA_MAX_TRY} times)")
        
        em = single_exact_match(pred, gold) if gold else 0.0
        f1 = single_f1_score(pred, gold) if gold else 0.0
        total_em += em
        total_f1 += f1
        n+=1
        print(f"\nquestion : {q}")
        print(f"gold_answer : {gold}")
        print(f"model_answer : {pred}")
        print(f"EM={em:.3f}, F1={f1:.3f}")          
        pred_list.append({
            "id":id,
            "question":q,
            "gold":gold,
            "pred":pred,
            "em":em,
            "f1":f1,
        })
        with open(res_path,"w") as f:
            json.dump(pred_list,f,indent=4)
            
    avg_em=total_em / n if n else 0.0
    avg_f1=total_f1 / n if n else 0.0
    metrics = {
        "count": n,
        "em": avg_em,
        "f1": avg_f1,
        "context_type": context_type,
        "model_name": model_name,
        "max_qa_items": max_qa_items,
        "res_path": res_path,
        "metric_path": metric_path,
    }
        
    with open(metric_path,"w") as f:
        json.dump(metrics,f,indent=4)

    print(f"[Done] [{n}/{len(data)}] EM={avg_em:.4f} F1={avg_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing HippoRAG")
    parser.add_argument('--bench_name', type=str, default="hotpotqa")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
    parser.add_argument('--context_type',type=str, default="no_context")
    parser.add_argument('--max_qa_items',type=int, default=100)
    args = parser.parse_args()
    
    bench_name = args.bench_name
    model_name=args.model_name
    context_type=args.context_type
    max_qa_items=args.max_qa_items
    
    if bench_name=="hotpotqa":
        CORPUS_PATH = HOTPOTQA_CORPUS_PATH
        EVAL_DATA_PATH = HOTPOTQA_QA_PATH
    elif bench_name=="musique":
        CORPUS_PATH = MUSIQUE_CORPUS_PATH
        EVAL_DATA_PATH = MUSIQUE_QA_PATH
    else:
        raise ValueError(f"Invalid bench_name: {bench_name}")
    if bench_name=="hotpotqa":
        CORPUS_PATH = HOTPOTQA_CORPUS_PATH
        EVAL_DATA_PATH = HOTPOTQA_QA_PATH
    elif bench_name=="musique":
        CORPUS_PATH = MUSIQUE_CORPUS_PATH
        EVAL_DATA_PATH = MUSIQUE_QA_PATH
    else:
        raise ValueError(f"Invalid bench_name: {bench_name}")
    RES_DIR=f"baseline_res_{bench_name}"
    os.makedirs(RES_DIR,exist_ok=True)

    llm_run_qa(bench_name,model_name,context_type,max_qa_items)
    
    
'''
export VLLM_QWEN_API_KEY="b1aed5830f0c80c1ef288e2b79122c847fd93a0c3cab5275c3dac9044877df39"
python vanilla_bsline.py --bench_name "hotpotqa" --model_name "Qwen2.5-32B-Instruct" --context_type "no_context" --max_qa_items 1000
python vanilla_bsline.py --bench_name "hotpotqa" --model_name "Qwen2.5-32B-Instruct" --context_type "gold_context" --max_qa_items 1000
'''