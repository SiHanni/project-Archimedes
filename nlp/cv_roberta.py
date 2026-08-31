"""
리뷰 태그 분류기 5-겹 교차검증 — KLUE-RoBERTa (TIPS 시험항목 1).

## 왜 교차검증인가

사람이 매긴 정답지가 100건뿐이다. 이걸 60/20/20 으로 한 번 나누면 테스트가
20건이라 **한 건만 틀려도 정밀도가 4~5%p 씩 움직인다.** 실제로 같은 모델이
검증 96.2% / 테스트 76.9% 로 갈렸다 — 실력 차이가 아니라 표본이 작아서 생긴
흔들림이다. 이 수치로 합격/불합격을 판정하면 운에 맡기는 것이 된다.

5-겹 교차검증은 100건을 다섯 조각으로 나눠, 매번 한 조각을 테스트로 쓰고
나머지로 학습한다. **모든 건이 정확히 한 번씩 테스트된다.** 그래서 결과가
분할 운에 좌우되지 않고, 겹 간 편차(표준편차)로 신뢰 구간도 함께 말할 수 있다.

## 데이터

- 사람 라벨 100건 — 사람이 모델 예측을 보지 않고 직접 분류. 평가는 **오직 이것**으로만 한다.
- 약라벨 273건 — Claude 예측을 교사로 쓰는 학습 보강분(지식 증류).
  ⚠️ 사람 라벨과 **같은 문장은 제외**했다(164건). 섞이면 테스트가 오염된다.
  약라벨은 **학습에만** 들어가고 평가에는 절대 쓰지 않는다.

## 판정 지표

답안지와 동일하게 센다 — 리뷰 × 태그 5종(해당없음 포함)을 하나씩 맞췄는지 보고
Precision = TP / (TP + FP).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAGS = ["KIND", "FAST", "PRICE", "NEGATIVE"]
MODEL_NAME = "klue/roberta-base"
MAX_LEN = 256
SEED = 42
FOLDS = 5
EPOCHS = 6

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ReviewSet(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            r["content"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
        )
        y = torch.tensor([1.0 if t in r["answer"] else 0.0 for t in TAGS])
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": y}


def folds_stratified(rows, k=FOLDS):
    """
    NEGATIVE 유무를 기준으로 층화해 k 조각으로 나눈다.

    NEGATIVE 가 5건뿐이라 무작위로 자르면 어떤 겹에는 0건이 되어 그 클래스의
    재현율이 정의되지 않는다. 층화하면 겹마다 1건씩 고르게 들어간다.
    """
    neg = [r for r in rows if "NEGATIVE" in r["answer"]]
    pos = [r for r in rows if "NEGATIVE" not in r["answer"]]
    random.shuffle(neg)
    random.shuffle(pos)
    out = [[] for _ in range(k)]
    for i, r in enumerate(neg):
        out[i % k].append(r)
    for i, r in enumerate(pos):
        out[i % k].append(r)
    return out


def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in loader:
            y = b.pop("labels")
            b = {k: v.to(device) for k, v in b.items()}
            P.append(model(**b).logits.sigmoid().cpu().numpy())
            Y.append(y.numpy())
    return np.vstack(P), np.vstack(Y) > 0.5


def score(P_bool, Y):
    """답안지와 동일한 집계 — 해당없음을 5번째 항목으로 포함한다."""
    none_p, none_y = ~P_bool.any(axis=1), ~Y.any(axis=1)
    tp = int((P_bool & Y).sum() + (none_p & none_y).sum())
    fp = int((P_bool & ~Y).sum() + (none_p & ~none_y).sum())
    return tp, fp, float((P_bool == Y).all(axis=1).mean())


def main():
    human = json.loads(Path("/tmp/tips/reviews100.json").read_text())
    weak = [
        {"content": r["content"], "answer": [t for t in (r["predicted_tags"] or "").split(",") if t]}
        for r in json.loads(Path("/tmp/tips/weak.json").read_text())
    ]
    parts = folds_stratified(human)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"사람 라벨 {len(human)}건 · 약라벨 {len(weak)}건 · {FOLDS}-겹 교차검증\n")
    allP, allY, rowres = [], [], []
    for f in range(FOLDS):
        test = parts[f]
        train = [r for j, p in enumerate(parts) if j != f for r in p] + weak
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(TAGS), problem_type="multi_label_classification"
        ).to(device)
        tl = DataLoader(ReviewSet(train, tok), batch_size=8, shuffle=True)
        el = DataLoader(ReviewSet(test, tok), batch_size=8)

        pos = np.array([sum(1 for r in train if t in r["answer"]) for t in TAGS], dtype=np.float32)
        pw = torch.tensor(
            np.clip((len(train) - pos) / np.maximum(pos, 1), 1.0, 20.0), dtype=torch.float32, device=device
        )
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        for _ in range(EPOCHS):
            model.train()
            for b in tl:
                b = {k: v.to(device) for k, v in b.items()}
                y = b.pop("labels")
                loss = lossf(model(**b).logits, y)
                loss.backward()
                opt.step()
                opt.zero_grad()

        P, Y = predict(model, el, device)
        # 임계값은 **고정 0.5**. 작은 테스트 겹에서 임계를 고르면 그 겹에 맞춰져
        # 성능이 부풀려진다(실측: 검증 96% / 테스트 77% 로 갈렸다).
        Pb = P > 0.5
        tp, fp, ex = score(Pb, Y)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rowres.append(prec)
        allP.append(Pb)
        allY.append(Y)
        print(f"  겹 {f+1}  n={len(test):2d}  Precision {prec:.4f}  완전일치 {ex:.2f}  (TP {tp} / FP {fp})")
        del model
        if device == "mps":
            torch.mps.empty_cache()

    P = np.vstack(allP)
    Y = np.vstack(allY)
    tp, fp, ex = score(P, Y)
    prec = tp / (tp + fp)
    print(f"\n=== 전체 100건 (각 건이 정확히 한 번씩 테스트됨) ===")
    print(f"Precision {prec:.4f}   완전일치 {ex:.3f}   TP {tp} / FP {fp}")
    print(f"겹별 편차: 평균 {np.mean(rowres):.4f} ± {np.std(rowres):.4f}  (최소 {min(rowres):.3f} · 최대 {max(rowres):.3f})")
    print(f"판정기준 0.873 — {'적합' if prec >= 0.873 else '부적합'}\n")

    per = {}
    for i, t in enumerate(TAGS):
        a = int((P[:, i] & Y[:, i]).sum())
        b = int((P[:, i] & ~Y[:, i]).sum())
        c = int((~P[:, i] & Y[:, i]).sum())
        per[t] = {"tp": a, "fp": b, "fn": c,
                  "precision": a / (a + b) if a + b else None,
                  "recall": a / (a + c) if a + c else None}
        pp = "―" if per[t]["precision"] is None else f"{per[t]['precision']:.3f}"
        rr = "―" if per[t]["recall"] is None else f"{per[t]['recall']:.3f}"
        print(f"  {t:9s} P {pp}  R {rr}   (tp {a} fp {b} fn {c})")

    Path(__file__).with_name("cv_metrics.json").write_text(
        json.dumps(
            {"precision": prec, "exact_match": ex, "tp": tp, "fp": fp,
             "fold_precisions": rowres, "per_tag": per,
             "n_human": len(human), "n_weak": len(weak), "folds": FOLDS, "epochs": EPOCHS},
            ensure_ascii=False, indent=1)
    )


if __name__ == "__main__":
    main()
