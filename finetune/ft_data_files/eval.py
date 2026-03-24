import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

# --- 1. 설정 (Configuration) ---
MODEL_NAME = "jjm0413/deekseek-coder-finetuned"
DATA_PATH = "/Users/jangjunmin/Documents/code/AI/gbsw_teamproject/Create_data/test_data.jsonl" # 평가할 로컬 파일명
K_VALUES = [1, 10]            # 계산할 pass@k 값

# ⚠️ 안전 경고: 모델이 생성한 코드를 실행하도록 허용
# 반드시 격리된 안전 환경에서 실행하세요!
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# --- 2. 코어 평가 함수 ---

def run_evaluation(model, tokenizer, eval_dataset, k_values):
    """모델의 pass@k 지표를 계산하고 결과를 반환합니다."""
    
    code_eval_metric = load("code_eval")
    max_k = max(k_values)
    
    predictions = []
    references = []
    
    print(f"총 {len(eval_dataset)}개 샘플 평가 시작 (각 샘플당 {max_k}개 생성)...")

    # 1. 코드 생성 루프
    for sample in tqdm(eval_dataset, desc="Generating & Collecting"):
        prompt = sample["prompt"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=max_k,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 생성된 텍스트 디코딩
        gen_texts = tokenizer.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        references.append(sample["test"])
        predictions.append(gen_texts)
        
    # 2. pass@k 계산 및 반환
    print("\n생성된 코드를 실행하여 pass@k 최종 계산 중...")
    pass_at_k_scores, _ = code_eval_metric.compute(
        references=references, 
        predictions=predictions, 
        k=k_values
    )
    
    return pass_at_k_scores

# --- 3. 메인 실행 흐름 ---

if __name__ == "__main__":
    
    print("="*50)
    print(f"'{MODEL_NAME}' 모델 평가 스크립트")
    print("="*50)

    try:
        # A. 모델 로드 (GPU 사용을 위해 device_map="auto" 사용)
        print("1. 모델 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        # B. 데이터셋 로드
        print(f"2. 데이터 파일 '{DATA_PATH}' 로드 중...")
        # JSON Lines 형식의 로컬 파일 로드 가정
        raw_dataset = load_dataset("json", data_files=DATA_PATH)
        eval_dataset = raw_dataset["train"]
        
        # ⚠️ 컬럼명 확인: 데이터셋에 "prompt"와 "test" 컬럼이 있는지 확인하세요.
        # 만약 컬럼명이 다르다면 아래와 같이 map 함수로 이름을 맞춰야 합니다.
        # eval_dataset = eval_dataset.map(lambda x: {"prompt": x["your_prompt_col"], "test": x["your_test_col"]})

        # C. 평가 실행
        print("3. 평가 실행 중...")
        pass_at_k_scores = run_evaluation(model, tokenizer, eval_dataset, K_VALUES)
        
        # D. 결과 출력
        print("\n" + "="*50)
        print("최종 평가 결과 (Pass@k)")
        print("="*50)
        for k, score in pass_at_k_scores.items():
            print(f"Pass@{k}: {score:.4f}")
        print("="*50)
        
    except FileNotFoundError:
        print(f"\n오류: 데이터 파일 '{DATA_PATH}'를 찾을 수 없습니다. 경로를 확인하세요.")
    except Exception as e:
        print(f"\n치명적인 오류 발생: {e}")