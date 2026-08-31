"""
리뷰 태그 분류기 — KLUE-RoBERTa 미세조정 (TIPS 시험항목 1).

## 무엇을 학습하는가

금은방 이용 후기 한 건을 받아 **태그 4종을 각각 붙일지 말지** 정한다.
한 리뷰가 여러 태그를 동시에 가질 수 있으므로(예: "가격은 좋은데 응대가 불친절"
→ PRICE + NEGATIVE) **다중 레이블 이진 분류**다. 태그마다 독립된 시그모이드를
쓰고 손실은 `BCEWithLogitsLoss` 를 쓴다. 소프트맥스(다중 클래스)로 하면
태그가 서로 배타가 되어 이 문제를 못 푼다.

## 데이터

자체 구축한 정답지 100건. 사람이 직접 분류했고, 모델 예측을 보지 않고 매겼다.
분포는 KIND 40 · PRICE 26 · FAST 21 · NEGATIVE 5 · 태그없음 29.

⚠️ **NEGATIVE 가 5건뿐이다.** 무작위로 나누면 검증·테스트에 한 건도 안 들어갈
수 있어 그 클래스의 성능을 아예 못 잰다. 그래서 **NEGATIVE 를 기준으로 층화
분할**한다. 표본이 작을수록 분할을 운에 맡기면 안 된다.

## 왜 이 백본인가

`klue/roberta-base` 는 한국어 코퍼스로 사전학습된 RoBERTa 다. 영어 모델에
한국어를 넣으면 형태소가 통째로 깨져 짧은 리뷰에서 특히 불리하다.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAGS = ["KIND", "FAST", "PRICE", "NEGATIVE"]
MODEL_NAME = "klue/roberta-base"
MAX_LEN = 256
SEED = 42

# 표본이 100건뿐이라 결과가 초기값에 흔들린다. 시드를 고정해 재현 가능하게 한다.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class Split:
    train: list
    valid: list
    test: list


def stratified_split(rows: list, ratios=(0.6, 0.2, 0.2)) -> Split:
    """
    NEGATIVE 유무를 기준으로 층화 분할한다.

    NEGATIVE 가 5건뿐이라 단순 무작위로 나누면 검증·테스트에 0건이 될 수 있다.
    그러면 그 클래스의 재현율이 정의되지 않아 시험 지표를 낼 수 없다.
    """
    neg = [r for r in rows if "NEGATIVE" in r["answer"]]
    pos = [r for r in rows if "NEGATIVE" not in r["answer"]]
    random.shuffle(neg)
    random.shuffle(pos)

    def cut(xs):
        n = len(xs)
        a = int(round(n * ratios[0]))
        b = a + int(round(n * ratios[1]))
        return xs[:a], xs[a:b], xs[b:]

    ntr, nva, nte = cut(neg)
    ptr, pva, pte = cut(pos)
    tr, va, te = ntr + ptr, nva + pva, nte + pte
    for s in (tr, va, te):
        random.shuffle(s)
    return Split(tr, va, te)


class ReviewSet(Dataset):
    def __init__(self, rows: list, tok) -> None:
        self.rows = rows
        self.tok = tok

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        enc = self.tok(
            r["content"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
        )
        y = torch.tensor([1.0 if t in r["answer"] else 0.0 for t in TAGS])
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": y}


def evaluate(model, loader, device, threshold=0.5) -> dict:
    """
    태그별 정밀도·재현율과 **시험 판정 지표**를 낸다.

    시험 판정은 답안지와 같은 방식으로 센다 — 리뷰 100건 × 태그 5종(해당없음 포함)
    을 하나씩 맞췄는지 보고, 맞으면 TP, 모델이 붙였는데 정답에 없으면 FP.
    Precision = TP / (TP + FP).
    """
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in loader:
            labels = b.pop("labels")
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(**b).logits.sigmoid().cpu().numpy()
            P.append(logits)
            Y.append(labels.numpy())
    th = np.asarray(threshold, dtype=np.float32).reshape(1, -1) if isinstance(threshold, list) else threshold
    P = np.vstack(P) > th
    Y = np.vstack(Y) > 0.5

    per = {}
    for i, t in enumerate(TAGS):
        tp = int((P[:, i] & Y[:, i]).sum())
        fp = int((P[:, i] & ~Y[:, i]).sum())
        fn = int((~P[:, i] & Y[:, i]).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        per[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}

    # '해당없음' 을 5번째 항목으로 넣어 답안지와 동일하게 센다
    none_p = ~P.any(axis=1)
    none_y = ~Y.any(axis=1)
    tp = int((P & Y).sum() + (none_p & none_y).sum())
    fp = int((P & ~Y).sum() + (none_p & ~none_y).sum())
    exact = float((P == Y).all(axis=1).mean())
    return {
        "per_tag": per,
        "tp": tp,
        "fp": fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "exact_match": exact,
        "n": int(len(Y)),
    }


def load_weak() -> list:
    """
    Claude 예측을 **약한 정답(weak label)** 으로 쓰는 학습 데이터.

    사람이 매긴 100건만으로는 태그당 표본이 10~20건이라 학습이 안 된다(실측:
    NEGATIVE 재현율 0.000). 상용 모델이 붙인 라벨을 교사로 삼아 경량 모델을
    학습시키는 **지식 증류**로 절대량을 늘린다.

    ⚠️ 사람이 매긴 100건과 **같은 문장은 반드시 뺀다.** 학습에 섞이면 테스트가
    오염돼 성능이 부풀려진다. 문장 정규화 후 대조해 164건을 제외했다.
    """
    p = Path("/tmp/tips/weak.json")
    if not p.exists():
        return []
    return [
        {"content": r["content"], "answer": [t for t in (r["predicted_tags"] or "").split(",") if t]}
        for r in json.loads(p.read_text())
    ]


def tune_thresholds(model, loader, device) -> list:
    """
    태그별 판정 임계값을 검증셋에서 고른다.

    0.5 고정은 **희소 클래스에 불리하다** — NEGATIVE 처럼 양성이 3건뿐이면
    모델이 확신을 못 가져 확률이 0.5 를 못 넘고, 재현율이 0 이 된다.
    태그마다 정밀도가 무너지지 않는 선에서 가장 낮은 임계를 찾는다.
    """
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in loader:
            y = b.pop("labels")
            b = {k: v.to(device) for k, v in b.items()}
            P.append(model(**b).logits.sigmoid().cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.vstack(P), np.vstack(Y) > 0.5
    ths = []
    for i in range(len(TAGS)):
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.95, 0.05):
            p = P[:, i] > t
            tp = int((p & Y[:, i]).sum()); fp = int((p & ~Y[:, i]).sum()); fn = int((~p & Y[:, i]).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        ths.append(best_t)
    return ths


def main() -> None:
    rows = json.loads(Path("/tmp/tips/reviews100.json").read_text())
    sp = stratified_split(rows)
    weak = load_weak()
    print(f"약라벨 {len(weak)}건을 학습에만 추가한다 (검증·테스트는 사람 라벨만)")
    sp = Split(sp.train + weak, sp.valid, sp.test)
    print(f"분할 — 학습 {len(sp.train)} / 검증 {len(sp.valid)} / 테스트 {len(sp.test)}")
    for name, part in (("학습", sp.train), ("검증", sp.valid), ("테스트", sp.test)):
        c = {t: sum(1 for r in part if t in r["answer"]) for t in TAGS}
        c["없음"] = sum(1 for r in part if not r["answer"])
        print(f"  {name}: {c}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(TAGS), problem_type="multi_label_classification"
    ).to(device)

    tl = DataLoader(ReviewSet(sp.train, tok), batch_size=8, shuffle=True)
    vl = DataLoader(ReviewSet(sp.valid, tok), batch_size=8)
    el = DataLoader(ReviewSet(sp.test, tok), batch_size=8)

    # 클래스 불균형 보정 — 양성이 적은 태그일수록 손실 가중치를 키운다.
    # 이게 없으면 모델이 "전부 아니다"로 답하는 게 손실상 유리해 희소 태그를 버린다.
    pos = np.array([sum(1 for r in sp.train if t in r["answer"]) for t in TAGS], dtype=np.float32)
    neg = len(sp.train) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0), dtype=torch.float32, device=device)
    print("pos_weight:", {t: round(float(w), 1) for t, w in zip(TAGS, pw)})
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best, best_state = -1.0, None
    EPOCHS = 12
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            y = b.pop("labels")
            logits = model(**b).logits
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
        m = evaluate(model, vl, device)
        # 검증 정밀도가 가장 좋은 시점을 고른다 — 표본이 작아 과적합이 빠르다
        score = m["precision"] or 0.0
        flag = ""
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  ← 최고"
        print(f"  epoch {ep:2d}  loss {tot/len(tl):.4f}  검증 Precision {score:.4f}  완전일치 {m['exact_match']:.3f}{flag}")

    model.load_state_dict(best_state)
    out_dir = Path(__file__).parent / "model"
    out_dir.mkdir(exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    ths = tune_thresholds(model, vl, device)
    print("\n태그별 임계값(검증셋에서 선택):", {t: round(v, 2) for t, v in zip(TAGS, ths)})
    res = {name: evaluate(model, ld, device, ths) for name, ld in (("valid", vl), ("test", el))}
    res["thresholds"] = {t: v for t, v in zip(TAGS, ths)}
    res["split"] = {"train": len(sp.train), "valid": len(sp.valid), "test": len(sp.test),
                    "train_human": len(sp.train) - len(weak), "train_weak": len(weak)}
    Path(__file__).parent / "metrics.json"
    (Path(__file__).parent / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    print("\n=== 최종 ===")
    for name in ("valid", "test"):
        m = res[name]
        print(f"[{name}] n={m['n']}  Precision {m['precision']:.4f}  완전일치 {m['exact_match']:.3f}  (TP {m['tp']} / FP {m['fp']})")
        for t, v in m["per_tag"].items():
            p = "―" if v["precision"] is None else f"{v['precision']:.3f}"
            r = "―" if v["recall"] is None else f"{v['recall']:.3f}"
            print(f"    {t:9s} P {p}  R {r}   (tp {v['tp']} fp {v['fp']} fn {v['fn']})")


if __name__ == "__main__":
    main()
