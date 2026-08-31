/**
 * 심사 제출용 **모델·데이터셋·연산과정 명세** 화면.
 *
 * 심사에서 요구한 항목을 한 페이지에 담는다 —
 *   ① 축별 모델 이름 · 학습 데이터셋 출처 · 학습/검증/테스트 개수 · 수집 기간
 *   ② 단계별 연산 과정
 *   ③ 실사진 10장의 픽셀 결과와 거리 결과
 *
 * ⚠️ **원 모델의 학습 수치와 우리 자체 수치를 반드시 구분해 적는다.**
 *    BiRefNet·Depth Pro 는 공개 사전학습 모델이라 그 학습 데이터는 원저작자의
 *    것이다. 우리가 학습시킨 것처럼 적으면 조회 한 번에 드러난다.
 *    우리 몫은 '자체 평가셋 구축 · 파라미터 실측 보정 · 후처리 신규 개발'이다.
 */
import { useEffect, useState } from 'react';

type Row = {
  gt_px: number; model_px: number; inter_px: number; miss_px: number; over_px: number;
  union_px: number; iou: number; dice: number; recall: number; precision: number;
  w: number; h: number; item: string; date: string; old_iou: number;
  dist: { k: string; cm: number; lo: number; hi: number; sig: number };
};

const KOR: Record<string, string> = {
  ring: '반지', necklace: '목걸이', earring: '귀걸이', goldbar: '골드바',
};

const ORDER = ['T379_01','T374_01','T390_01','T332_01','T338_01','T384_01','T341_01','T192_01','T330_01','T152_01'];

function Num({ v }: { v: number }) {
  return <>{v.toLocaleString()}</>;
}

export default function SpecPage() {
  const [rows, setRows] = useState<Record<string, Row> | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    fetch('/spec/outline.json').then((r) => r.json()).then(setRows).catch(() => setRows({}));
  }, []);

  const ids = rows ? ORDER.filter((k) => rows[k]) : [];
  const mIoU = ids.length ? ids.reduce((a, k) => a + rows![k].iou, 0) / ids.length : 0;
  const oldIoU = ids.length ? ids.reduce((a, k) => a + rows![k].old_iou, 0) / ids.length : 0;

  return (
    <div className="spec">
      <h2>모델 · 데이터셋 · 연산과정 명세</h2>
      <p className="muted">
        연구개발 과제 시험항목 4종에 대한 사용 모델, 학습 데이터 출처, 단계별 연산 과정과
        실사진 10건의 측정 결과입니다. 모든 수치는 실측값입니다.
      </p>

      {/* ── 요약 ── */}
      <div className="notice">
        <p className="notice__title">한눈에</p>
        <div className="table-wrap">
          <table className="batch-table">
            <thead>
              <tr><th>시험항목</th><th>사용 모델</th><th>유형</th><th className="n">자체 측정값</th><th>기준</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>1. 자연어 처리 정확도</td>
                <td>Claude Opus 5</td>
                <td>상용 API · 제로샷</td>
                <td className="n strong">92.56%</td>
                <td>Precision ≥ 87.3%</td>
              </tr>
              <tr>
                <td>2. 세미-오토 라벨링</td>
                <td>BiRefNet + 자체 후처리</td>
                <td>사전학습 + 자체 보정</td>
                <td className="n strong">{mIoU.toFixed(2)}%</td>
                <td>mIoU ≥ 70%</td>
              </tr>
              <tr>
                <td>3. 거리 추정</td>
                <td>Depth Pro + 핀홀 기하</td>
                <td>사전학습 + 자체 산식</td>
                <td className="n">범위 제시</td>
                <td className="dim">별도 기준 없음</td>
              </tr>
              <tr>
                <td>4. 매칭 점수</td>
                <td className="dim">해당 없음</td>
                <td><b>규칙 기반</b> (학습 없음)</td>
                <td className="n">RMSE 0</td>
                <td>RMSE ≤ 0.52</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ── 축 1 ── */}
      <h3>① 자연어 처리 — 리뷰 태그 분류</h3>
      <div className="table-wrap">
        <table className="batch-table spec-kv">
          <tbody>
            <tr><th>운영 모델</th><td><code>claude-opus-5</code> (Anthropic API)</td></tr>
            <tr><th>방식</th><td><b>제로샷 분류</b> — 미세조정 없이 지시문(프롬프트)만으로 분류. 따라서 <b>학습 데이터 0건</b>.</td></tr>
            <tr><th>사전학습 데이터</th><td className="dim">모델 제작사(Anthropic) 비공개</td></tr>
            <tr><th>자체 평가셋</th><td><b>리뷰 100건</b> — 사람이 직접 분류(모델 예측 비공개 상태에서 라벨링). 출처: 실제 유저 리뷰 · 시드 리뷰 · 부정 표본 확보용 작성 리뷰</td></tr>
            <tr><th>수집 기간</th><td><b>2026-03-09 ~ 2026-08-31</b> (실제 유저 리뷰 88건 기준)</td></tr>
            <tr><th>태그 분포</th><td>친절 40 · 가격 26 · 신속 21 · <b>부정 5</b> · 해당없음 29</td></tr>
            <tr><th>측정값</th><td><b>Precision 92.56%</b> (기준 87.3% 이상 — 적합)</td></tr>
          </tbody>
        </table>
      </div>

      <div className="notice notice--tips">
        <p className="notice__title">자체 모델 학습 시도 — 채택하지 않음</p>
        <p style={{ margin: 0 }}>
          외부 API 의존을 줄이기 위해 <b>한국어 RoBERTa(<code>klue/roberta-base</code>)를 직접 미세조정</b>했습니다.
          자체 정답지 100건에, 상용 모델 예측을 약한 정답으로 쓰는 <b>지식 증류</b> 273건을 더해
          학습했고, 표본이 작아 결과가 분할 운에 흔들리지 않도록 <b>5겹 교차검증</b>으로 측정했습니다.
        </p>
        <div className="table-wrap" style={{ marginTop: '0.6rem' }}>
          <table className="batch-table">
            <thead><tr><th>구분</th><th className="n">건수</th><th>비고</th></tr></thead>
            <tbody>
              <tr><td>사람 라벨 (평가 전용)</td><td className="n">100</td><td>5겹 교차검증 — 모든 건이 정확히 한 번 테스트됨</td></tr>
              <tr><td>약라벨 (학습 보강)</td><td className="n">273</td><td>사람 라벨과 겹치는 164건은 제외 — 테스트 오염 방지</td></tr>
              <tr><td>겹당 학습 / 테스트</td><td className="n">353 / 20</td><td>부정 클래스 기준 층화 분할</td></tr>
            </tbody>
          </table>
        </div>
        <p style={{ margin: '0.6rem 0 0' }}>
          결과는 <b>Precision 79.03%</b>(겹별 79.2% ± 7.1%p)로 기준 87.3%에 미달했습니다.
          원인은 <b>부정 리뷰가 전체에서 5건뿐</b>이라는 데이터 부족이며, 클래스 가중치 20배와
          임계값 조정으로도 부정 태그 재현율이 0에 머물렀습니다. 이에 따라 운영 모델로는
          상용 API를 채택했고, 자체 모델은 부정 표본이 확보되는 대로 재시도합니다.
        </p>
      </div>

      <h4>단계별 연산 과정</h4>
      <ol className="steps">
        <li><b>입력</b> — 리뷰 본문 1건. 공백 제외 10자 미만은 제외(별점만 남긴 리뷰)</li>
        <li><b>지시문 구성</b> — 태그 4종의 정의와 판정 규칙 4개를 시스템 지시문으로 제시</li>
        <li><b>추론</b> — 출력 형식을 <code>{'{tags:[...], reason:"..."}'}</code> 로 강제(구조화 출력). 강제하지 않으면 설명 문장이 섞여 파싱이 깨진다</li>
        <li><b>정규화</b> — 태그를 <b>알파벳 오름차순 CSV</b>로 저장. 정렬을 강제해야 문자열 비교만으로 정오 판정이 된다</li>
        <li><b>채점</b> — 리뷰 × 태그 5종(해당없음 포함)을 하나씩 대조. 맞으면 TP, 모델이 붙였는데 정답에 없으면 FP. <code>Precision = TP / (TP+FP)</code></li>
      </ol>

      {/* ── 축 2 ── */}
      <h3>② 세미-오토 라벨링 — 외곽선 추출</h3>
      <div className="table-wrap">
        <table className="batch-table spec-kv">
          <tbody>
            <tr><th>모델</th><td><b>BiRefNet</b> — <code>onnx-community/BiRefNet-ONNX</code>, 백본 Swin Transformer</td></tr>
            <tr><th>논문</th><td>Bilateral Reference for High-Resolution Dichotomous Image Segmentation (CAAI AIR, 2024)</td></tr>
            <tr><th>라이선스</th><td><b>MIT</b> — 상업적 이용 제약 없음</td></tr>
            <tr><th>사전학습 데이터셋<br /><span className="dim">(원저작자)</span></th><td>
              <b>DIS5K</b> — 고해상도 이분할 데이터셋. Flickr 수집, 22개 군 225개 범주, 총 <b>5,470장</b><br />
              <span className="dim">학습(DIS-TR) 3,000 · 검증(DIS-VD) 470 · 테스트(DIS-TE) 2,000</span><br />
              범용 가중치는 DUTS · HRSOD · UHRSD · P3M 등을 함께 사용
            </td></tr>
            <tr><th>모델 선정 근거</th><td>후보 2종을 <b>동일 표본 10장</b>으로 비교 — RMBG-1.4는 10장 중 5장 실패(저울 전체 66.8% · 화면 전체 99.99%), BiRefNet은 10장 전부 물체만 분리</td></tr>
            <tr><th>자체 평가셋</th><td><b>실거래 사진 10건</b> · 수작업 정답 마스크 대조 · 수집 기간 <b>2026-05-14 ~ 2026-08-04</b></td></tr>
            <tr><th>측정값</th><td><b>mIoU {mIoU.toFixed(2)}%</b> (기준 70% 이상 — 적합) · 종전 {oldIoU.toFixed(2)}% 대비 <b>+{(mIoU - oldIoU).toFixed(2)}%p</b></td></tr>
          </tbody>
        </table>
      </div>

      <h4>단계별 연산 과정</h4>
      <ol className="steps">
        <li><b>해상도 확인</b> — 짧은 변 240px 미만 거절. 이 경로는 크기를 주장하지 않으므로 문턱이 낮다</li>
        <li><b>모델 추론</b> — 1024×1024 입력, ImageNet 정규화 <code>(x/255 − μ)/σ</code>.
            출력은 <b>로짓</b>이므로 시그모이드 <code>p = 1/(1+e⁻ᶻ)</code> 를 거쳐 0.5로 이진화.
            <span className="dim"> ※ 최소·최대 정규화를 쓰면 물체 없는 사진에서 잡음이 1.0까지 증폭돼 화면 전체가 물체가 된다</span></li>
        <li><b>크롭 재추론</b> — 1차 결과의 테두리 상자(여백 6%)로 잘라 <b>같은 모델에 재투입</b>.
            물체가 화면을 크게 차지해 경계를 다시 정밀하게 잡는다. 채택 조건 <code>0.2 ≤ 2차/1차 ≤ 1.3</code></li>
        <li><b>비귀금속 덩어리 제외</b> — 덩어리 단위로 <code>채도중앙 ≥ 0.20 <b>또는</b> 정반사비율 ≥ 0.005</code>.
            <span className="dim"> ※ ‘또는’이어야 한다. ‘그리고’로 바꾸면 채도가 낮은 은·백금이 통째로 배경이 된다</span></li>
        <li><b>포장 파내기</b> — 색상(H)으로 가른다. 금 10~30 / 분홍 상자 160~180 으로 겹치지 않는다.
            모폴로지 열기로 금속 하이라이트를 걸러내고, 덩어리 정반사비율 ≥ 0.02 면 금속으로 보아 보존</li>
        <li><b>구멍 파내기</b> — 반지 구멍처럼 배경이 비치는 영역을 제거. 물체 둘레 배경띠의 Lab 중앙값과
            거리 ≤ 45 이면 ‘비쳐 보이는 배경’으로 판정</li>
        <li><b>산출</b> — 외곽선 오버레이 · 이진 마스크(원본 해상도) · 배경 투명 누끼 ·
            폴리곤 좌표(바깥선 + 구멍, <code>ε = 0.002 × 둘레</code>)</li>
      </ol>

      <h4>실사진 10건 측정 결과</h4>
      <p className="muted small">
        IoU = 교집합 ÷ 합집합 · 재현율 = 교집합 ÷ 정답 · 정밀도 = 교집합 ÷ 모델.
        행을 누르면 원본 · 추출 결과 · 대조 이미지가 열립니다.
      </p>
      <div className="table-wrap">
        <table className="batch-table">
          <thead>
            <tr>
              <th>이미지</th><th>품목</th><th className="n">정답(px)</th><th className="n">모델(px)</th>
              <th className="n">교집합</th><th className="n">누락</th><th className="n">과다</th>
              <th className="n">IoU</th><th className="n">종전</th><th className="n">거리</th>
            </tr>
          </thead>
          <tbody>
            {ids.map((k) => {
              const r = rows![k];
              return (
                <>
                  <tr key={k} className="clickable" onClick={() => setOpen(open === k ? null : k)}>
                    <td><b>{k.replace('_01', '')}</b> <span className="dim">{open === k ? '▾' : '▸'}</span></td>
                    <td className="fname">{r.item}</td>
                    <td className="n"><Num v={r.gt_px} /></td>
                    <td className="n"><Num v={r.model_px} /></td>
                    <td className="n"><Num v={r.inter_px} /></td>
                    <td className="n">{r.miss_px.toLocaleString()}</td>
                    <td className="n">{r.over_px.toLocaleString()}</td>
                    <td className="n strong">{r.iou.toFixed(1)}%</td>
                    <td className="n dim">{r.old_iou.toFixed(1)}%</td>
                    <td className="n">{r.dist.cm} cm</td>
                  </tr>
                  {open === k ? (
                    <tr key={`${k}-d`}>
                      <td colSpan={10}>
                        <div className="shots">
                          {[['orig', '원본 (모델 입력)'], ['overlay', '모델 추출 영역'], ['diff', '정답 대조']].map(
                            ([suf, label]) => (
                              <figure key={suf}>
                                <img src={`/spec/${k}_${suf}.jpg`} alt={label} loading="lazy" />
                                <figcaption>{label}</figcaption>
                              </figure>
                            ),
                          )}
                        </div>
                        <p className="muted small" style={{ marginTop: '0.4rem' }}>
                          대조 — <span style={{ color: '#3caa3c' }}>■ 초록 일치</span>{' '}
                          <span style={{ color: '#dc3c3c' }}>■ 빨강 누락(정답만)</span>{' '}
                          <span style={{ color: '#3c78dc' }}>■ 파랑 과다(모델만)</span>
                          {' · '}재현율 {r.recall.toFixed(1)}% · 정밀도 {r.precision.toFixed(1)}% ·
                          Dice {r.dice.toFixed(1)}% · 원본 {r.w}×{r.h} · 거래일 {r.date}
                        </p>
                      </td>
                    </tr>
                  ) : null}
                </>
              );
            })}
            <tr className="sum">
              <td colSpan={7}><b>평균 mIoU</b></td>
              <td className="n strong">{mIoU.toFixed(2)}%</td>
              <td className="n dim">{oldIoU.toFixed(2)}%</td>
              <td className="n dim">—</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── 축 3 ── */}
      <h3>③ 거리 추정 <span className="tag-aux">시험항목 외 · 부가 기능</span></h3>
      <div className="notice">
        <p className="notice__title">시험항목과의 관계</p>
        <p style={{ margin: 0 }}>
          이 항목은 <b>완료판정 시험 대상이 아닙니다.</b> 시험항목 4(매칭)에 나오는 ‘거리’와는
          다른 값이므로 혼동하지 않도록 구분해 둡니다.
        </p>
        <div className="table-wrap" style={{ marginTop: '0.6rem' }}>
          <table className="batch-table">
            <thead><tr><th>구분</th><th>시험항목 4 의 거리</th><th>본 항목의 거리</th></tr></thead>
            <tbody>
              <tr><td>무엇을 재나</td><td>고객 ↔ 금은방</td><td><b>카메라 ↔ 귀금속</b></td></tr>
              <tr><td>단위·범위</td><td>km (7.27 ~ 47.91)</td><td>cm (6.2 ~ 20.8)</td></tr>
              <tr><td>산출 방법</td><td>좌표 하버사인 직선거리</td><td>핀홀 기하 + 크기 가정</td></tr>
              <tr><td>정답 유무</td><td className="ok">있음 (좌표로 확정)</td><td className="bad">없음 (실측 표본 미확보)</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div className="table-wrap">
        <table className="batch-table spec-kv">
          <tbody>
            <tr><th>모델</th><td><b>Apple Depth Pro</b> (ONNX) — <b>초점거리 추정에만</b> 사용</td></tr>
            <tr><th>논문</th><td>Depth Pro: Sharp Monocular Metric Depth in Less Than a Second (2024)</td></tr>
            <tr><th>라이선스</th><td>Apple AMLR</td></tr>
            <tr><th>사전학습 데이터셋<br /><span className="dim">(원저작자)</span></th><td>
              실제 + 합성 데이터셋 혼합 학습(2단계 학습: 1단계 전체 라벨 데이터, 2단계 합성 데이터).
              개별 데이터셋 명세는 논문 부록에 수록.<br />
              <span className="dim">논문이 명시한 제로샷 평가 벤치마크: Booster · ETH3D · Middlebury · nuScenes · Sintel · Sun-RGBD
              (경계 정확도: iBims · AM-2k · P3M-10k · DIS-5k · Spring)</span>
            </td></tr>
            <tr><th>자체 평가셋</th><td>동일 실거래 사진 10건</td></tr>
            <tr><th>⚠️ 한계</th><td><b>정답 거리를 실측한 표본이 없어 정확도는 산출하지 않았습니다.</b> 제시값은 계산값과 그 불확실성 범위입니다.</td></tr>
          </tbody>
        </table>
      </div>

      <h4>산식</h4>
      <div className="eq-box">
        <div className="eq-main">Z = f × S / p</div>
        <div className="eq-sub">
          Z 거리[mm] · f 초점거리[px] · S 물체 실제 크기[mm] · p 사진 속 물체 크기[px]
        </div>
        <p className="muted small" style={{ marginTop: '0.6rem', marginBottom: 0 }}>
          핀홀 카메라에서 물체 쪽 삼각형(밑변 S · 높이 Z)과 센서 쪽 삼각형(밑변 p · 높이 f)이
          닮은꼴이므로 <code>p : f = S : Z</code> 가 성립합니다.
        </p>
      </div>

      <div className="notice warn-box">
        <p className="notice__title">왜 크기를 가정해야 하는가 — 단안 스케일 모호성</p>
        <p style={{ margin: 0 }}>
          사진에서 직접 아는 값은 <b>p 하나</b>인데 미지수는 <b>S와 Z 둘</b>입니다. 방정식이 하나 부족해
          수학적으로 풀리지 않습니다. S를 2배, Z를 2배 하면 p는 변하지 않아 <b>사진이 완전히 동일</b>합니다.
          즉 <b>“작은 물체가 가까이”와 “큰 물체가 멀리”를 원리적으로 구분할 수 없습니다.</b>
          알고리즘의 한계가 아니라 정보가 없는 것입니다.
        </p>
        <p style={{ margin: '0.5rem 0 0' }}>
          따라서 둘 중 하나를 외부에서 넣어야 합니다 — <b>크기를 가정</b>하면 거리가 나오고(본 항목),
          <b>신용카드(ISO/IEC 7810 ID-1 · 85.60×53.98mm)를 함께 촬영</b>하면 크기가 실측됩니다(중량·시세 산출 경로).
        </p>
      </div>

      <h4>단계별 연산 과정</h4>
      <ol className="steps">
        <li><b>외곽선 추출</b> — 위 ② 과정을 그대로 수행. 거리 계산의 선행 조건</li>
        <li><b>p 측정</b> — 마스크의 <b>최소외접사각형 긴 변</b>.
            <span className="dim"> ※ 축에 나란한 사각형(bbox)을 쓰면 물체가 비스듬할 때 대각선까지 감싸 p가 커지고 거리가 실제보다 가깝게 나온다</span></li>
        <li><b>f 확보</b> — EXIF에 촬영 정보가 있으면 실측값 사용, 없으면 Depth Pro 추정.
            입력은 <b>가로세로 비를 유지한 채 여백을 채워(레터박스)</b> 정사각으로 만든다</li>
        <li><b>S 확보</b> — 제품 종류의 대표 크기. 사용자가 실제 크기를 입력하면 그 값을 우선</li>
        <li><b>계산 · 검증</b> — <code>Z = f·S/p</code>. 40mm ~ 1,500mm 범위를 벗어나면 값을 내지 않는다
            <span className="dim"> (4cm 미만은 폰이 초점을 못 잡는 거리, 1.5m 초과는 귀금속 촬영 거리가 아님)</span></li>
        <li><b>오차 합성</b> — 크기 오차와 초점거리 오차는 독립이므로
            <code>√(크기폭² + 0.08²)</code>. 결과는 점이 아니라 <b>범위</b>로 제시</li>
      </ol>

      <div className="table-wrap">
        <table className="batch-table">
          <thead>
            <tr><th>제품</th><th className="n">대표 크기</th><th className="n">가정 폭</th><th className="n">최종 오차</th><th>근거</th></tr>
          </thead>
          <tbody>
            <tr><td>반지</td><td className="n">21 mm</td><td className="n">±12%</td><td className="n strong">±14%</td><td>손가락에 맞아야 해 안지름 15~22mm 로 묶임</td></tr>
            <tr><td>팔찌</td><td className="n">60 mm</td><td className="n">±25%</td><td className="n">±26%</td><td>손목 둘레 기준</td></tr>
            <tr><td>골드바</td><td className="n">42 mm</td><td className="n">±30%</td><td className="n">±31%</td><td>0.05g 카드형 ~ 100g 바</td></tr>
            <tr><td>귀걸이</td><td className="n">20 mm</td><td className="n">±40%</td><td className="n">±41%</td><td>형태 편차가 큼</td></tr>
            <tr><td>목걸이</td><td className="n">45 mm</td><td className="n">±45%</td><td className="n">±46%</td><td>펜던트·체인이 뭉친 정도에 따라</td></tr>
            <tr className="ok-row"><td>사용자 입력</td><td className="n">입력값</td><td className="n">±2%</td><td className="n strong">±8%</td><td>실측값이므로 초점거리 오차만 남음</td></tr>
          </tbody>
        </table>
      </div>

      <h4>실사진 10건 거리 결과</h4>
      <div className="table-wrap">
        <table className="batch-table">
          <thead>
            <tr><th>이미지</th><th>가정 품목</th><th className="n">추정 거리</th><th className="n">범위</th><th className="n">오차</th></tr>
          </thead>
          <tbody>
            {ids.map((k) => {
              const d = rows![k].dist;
              return (
                <tr key={k}>
                  <td><b>{k.replace('_01', '')}</b></td>
                  <td>{KOR[d.k] ?? d.k}</td>
                  <td className="n strong">{d.cm} cm</td>
                  <td className="n">{d.lo}~{d.hi} cm</td>
                  <td className="n">±{d.sig}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="notice notice--tips">
        <p className="notice__title">개발 중 발견·수정한 결함</p>
        <p style={{ margin: 0 }}>
          입력 사진을 가로세로 비를 무시하고 정사각형으로 <b>찌그러뜨려</b> 모델에 넣고 있었습니다.
          장면이 왜곡되면 모델이 화각을 실제보다 넓게 판단하고, 그만큼 초점거리가 작아져
          <b>거리가 짧게 계산</b>됩니다. 여백을 채우는 방식으로 바꾼 뒤 초점거리가 최대 +51% 증가했습니다.
        </p>
        <p style={{ margin: '0.5rem 0 0' }}>
          <b>진단의 근거</b> — 가로세로 비가 <b>1:1인 사진 2장만 값이 정확히 동일</b>했습니다
          (정사각형은 찌그러뜨려도 변형이 없음). 그리고 <b>세로로 가장 긴 사진(288×512)이 +51%로 가장 크게</b>
          변해, 왜곡이 클수록 오차가 크다는 예측과 일치했습니다.
        </p>
      </div>

      {/* ── 축 4 ── */}
      <h3>④ 매칭 점수</h3>
      <div className="notice warn-box">
        <p className="notice__title">이 항목은 머신러닝 모델이 아닙니다</p>
        <p style={{ margin: 0 }}>
          <b>규칙 기반 점수 산출(rule-based scoring)</b>입니다. 학습 과정이 없으므로
          학습 데이터셋·학습/검증/테스트 분할이 존재하지 않습니다. 사용하는 것은
          공개된 좌표 거리 공식과 사전에 합의한 점수 구간뿐입니다.
        </p>
      </div>

      <div className="eq-box">
        <div className="eq-main">매칭점수 = 거리점수 + 평점점수 &nbsp;(최대 6점)</div>
        <div className="eq-sub" style={{ marginTop: '0.5rem' }}>
          거리점수 — 3km 이내 3점 · 10km 이내 2점 · 10km 초과 1점<br />
          평점점수 — 4.8 이상 3점 · 4.0~4.7 2점 · 3.9 이하 1점
        </div>
      </div>

      <h4>단계별 연산 과정</h4>
      <ol className="steps">
        <li><b>거리 계산</b> — 매칭 위치와 파트너 좌표로 <b>하버사인(Haversine) 직선거리</b> 산출. 지구 반지름 6,371km</li>
        <li><b>구간 판정</b> — 위 표의 경계로 거리점수·평점점수를 각각 1~3점으로 환산</li>
        <li><b>합산</b> — 두 점수를 더해 매칭점수(2~6점)</li>
        <li><b>대조</b> — 수기 정답 점수와 비교해 RMSE 산출</li>
      </ol>

      <div className="notice">
        <p className="notice__title">RMSE 0 의 해석 — 반드시 함께 밝힐 것</p>
        <p style={{ margin: 0 }}>
          측정된 RMSE는 <b>0</b>입니다. 다만 이것은 정답 점수를 <b>같은 규칙으로 산출</b>했기 때문이며,
          예측 성능이 완벽하다는 뜻이 아닙니다. 규칙이 결정론적이므로 동일 입력에 동일 출력이 나오는
          것이 당연합니다. 이 항목이 검증하는 것은 <b>구현이 합의된 규칙을 정확히 따르는가</b>입니다.
        </p>
        <p style={{ margin: '0.5rem 0 0' }}>
          <b>현행 운영 매칭은 전국 브로드캐스트 + 선착순 자율 입찰</b>이라 점수가 배정에 관여하지 않습니다.
          타당성 참고 지표로 ‘점수 1위 파트너 vs 실제 낙찰 파트너’ 일치 여부를 함께 봅니다.
        </p>
      </div>

      {/* ── 정정 ── */}
      <h3>기재 정정 사항</h3>
      <div className="table-wrap">
        <table className="batch-table">
          <thead><tr><th>항목</th><th>종전 기재</th><th>실제 구현</th><th>비고</th></tr></thead>
          <tbody>
            <tr>
              <td>시험항목 1 모델명</td>
              <td className="dim">RoBERTa 기반</td>
              <td><b>Claude Opus 5</b> (Anthropic API)</td>
              <td>측정값 92.56%는 이 모델의 결과이므로 <b>수치는 그대로 유효</b>. 모델명만 정정</td>
            </tr>
            <tr>
              <td>시험항목 4 성격</td>
              <td className="dim">매칭 AI 모델</td>
              <td><b>규칙 기반 점수 산출</b></td>
              <td>학습 과정 없음. 데이터셋·분할 미해당</td>
            </tr>
            <tr>
              <td>시험항목 2 측정값</td>
              <td className="dim">mIoU {oldIoU.toFixed(2)}%</td>
              <td><b>mIoU {mIoU.toFixed(2)}%</b></td>
              <td>후처리 개선 반영. 동일 정답 마스크·동일 산식으로 재측정</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="muted small" style={{ marginTop: '2rem' }}>
        본 문서의 자체 측정값은 모두 실측입니다. 사전학습 모델의 학습 데이터 수치는 원저작자가 공개한 값을
        인용한 것이며, 본 과제에서 해당 모델을 재학습하지 않았습니다.
      </p>
    </div>
  );
}
