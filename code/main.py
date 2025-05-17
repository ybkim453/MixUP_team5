import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.experiment import ExperimentRunner

def main():
    load_dotenv()
    api_key = "up_IHiA1eseSoupjEiwKsbKzs3DXdh1T"
    if not api_key:
        raise ValueError("API key not found")

    template_name = "basic"
    config = ExperimentConfig(template_name=template_name, toy_size=100)

    train = pd.read_csv(os.path.join(config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(config.data_dir, 'test.csv'))
    toy_data = train.sample(n=config.toy_size, random_state=config.random_seed).reset_index(drop=True)

    # 전체 toy_data를 train/valid로 나누지 않고 단일 실험 실행
    print(f"\n=== '{template_name}' 템플릿 단일 성능 실험 시작 ===")

    runner = ExperimentRunner(config, api_key)
    result = runner.run_template_experiment(toy_data, toy_data)

    print("\n=== 단일 실험 결과 ===")
    print(f"  Train Recall: {result['train_recall']['recall']:.2f}%")
    print(f"  Train Precision: {result['train_recall']['precision']:.2f}%")
    print(f"  Valid Recall: {result['valid_recall']['recall']:.2f}%")
    print(f"  Valid Precision: {result['valid_recall']['precision']:.2f}%")
    # 제출 파일 생성 (단일 실행)
    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name=template_name,
        temperature=0.0,
        batch_size=5,
        experiment_name="final_submission"
    )
    runner = ExperimentRunner(config, api_key)
    test_results = runner.run(test)

    if isinstance(test_results, list):
        output = pd.DataFrame(test_results)
    else:
        output = pd.DataFrame({
            'id': test['id'],
            'cor_sentence': test_results['cor_sentence']
        })

    output.to_csv("submission5_cv.csv", index=False)
    print("\n✅ 제출 파일이 생성되었습니다: submission5_cv.csv")
    print(f"예측된 샘플 수: {len(output)}")
if __name__ == "__main__":
    main()
