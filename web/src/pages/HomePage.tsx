import { useCallback, useState } from 'react';
import { getJob, isQuoteSuppressed, postJob, type JobDto } from '../api';
import { CaptureGuide } from '../components/CaptureGuide';

const VIEWS = ['front', 'top', 'left', 'right', 'back'] as const;
const VIEW_LABEL: Record<(typeof VIEWS)[number], string> = {
  front: '정면',
  top: '상단',
  left: '좌측',
  right: '우측',
  back: '후면',
};

type CaptureMode = 'camera' | 'upload';
/** 사진 장수 — v2 는 1장이 기본, 5방향은 고신뢰 옵션 */
type ShotMode = 'single' | 'multiview';

/** worker `constants.MATERIALS` 와 동일해야 함 */
const METALS = [
  { value: 'gold', label: '금' },
  { value: 'silver', label: '은' },
  { value: 'platinum', label: '백금' },
] as const;

/** 함량 표기는 금속마다 의미가 달라(예: 999 = 금 24K vs 은 순은) 금속별로 나눈다 */
const PURITY_BY_METAL: Record<string, { value: string; label: string }[]> = {
  gold: [
    { value: '24k', label: '24K (순금)' },
    { value: '22k', label: '22K' },
    { value: '18k', label: '18K' },
    { value: '14k', label: '14K' },
    { value: '10k', label: '10K' },
  ],
  silver: [
    { value: 'sterling', label: '925 (실버)' },
    { value: 'fine', label: '999 (순은)' },
  ],
  platinum: [
    { value: 'pt950', label: 'Pt950' },
    { value: 'pt900', label: 'Pt900' },
    { value: 'pt999', label: 'Pt999 (순백금)' },
  ],
};

/** 형태별 촬영 주의 — API `product_k` 와 동일 키 (`worker` constants·스펙 §4·§7 정합) */
const PRODUCT_SHOOTING_TIPS: Record<string, { title: string; items: string[] }> = {
  plated: {
    title: '도금 · 금박 · 기념품',
    items: [
      '이런 제품은 **사진으로 금 무게를 잴 수 없습니다.** 몸체는 수지·황동이고 금은 마이크로미터 두께 막이라, 크기를 아무리 정확히 재도 금 함량과 무관합니다.',
      '예: "순금 0.005g" 기념 골드바 → 금 0.005g 을 펴면 두께가 **0.26μm**(금박)입니다. 몸체를 순금으로 치면 실제와 수백 배 차이가 납니다.',
      '실제 함유량은 **제품 표기**를 따라 주세요. 여기서는 실측 치수만 보여 드립니다.',
    ],
  },
  goldbar: {
    title: '골드바 · 골드카드',
    items: [
      '**케이스·블리스터에서 꺼내** 주세요. 투명 케이스도 그대로 재어 버립니다.',
      '두께는 사진으로 못 잽니다 — 각인이나 제품 규격의 **두께(mm)를 입력**해 주세요. 넓이는 사진에서 정확히 잽니다.',
      '카드 바로 옆에 **평평하게** 놓고, 화면에 크게 나오도록 가까이 찍어 주세요.',
    ],
  },
  ring: {
    title: '반지',
    items: [
      '반지와 카드를 **같은 바닥 면**에 두고, 링이 세워지지 않게 **눕힌 상태**로 촬영하면 스케일이 안정적입니다.',
      '속이 비어 있는 링·텅 빈 디자인은 실제 금속 질량과 추정치 차이가 클 수 있습니다.',
    ],
  },
  necklace: {
    title: '목걸이',
    items: [
      '체인을 **최대한 한 줄로 펼쳐**, 고리가 **겹치지 않게** 바닥에 놓고 촬영해 주세요. 뭉치면 실루엣이 한 덩어리로 잡혀 부피·질량 오차가 커질 수 있습니다.',
      '상단 뷰에서 **카드와 목걸이가 함께** 프레임에 들어오도록 해 주세요.',
    ],
  },
  chain: {
    title: '체인',
    items: [
      '체인을 **펼쳐** 한 줄로, **겹침을 최소화**해 바닥에 놓고 촬영해 주세요.',
      '일부만 뭉쳐 두면 안 됩니다. 가능한 범위에서 펼쳐 주세요.',
    ],
  },
  bracelet: {
    title: '팔찌',
    items: [
      '얇은 체인형 팔찌는 목걸이와 같이 **펼쳐** 겹침을 줄여 주세요.',
      '굵은 뱅글형도 카드와 **같은 바닥 면**에 두고, 찌그러지지 않게 촬영해 주세요.',
    ],
  },
  pendant: {
    title: '펜던트',
    items: [
      '펜던트 **본체**와 카드가 **같은 평면(바닥)** 에 놓이게 해 주세요.',
      '끈·체인은 가능하면 **펼쳐** 겹침을 줄여 주세요.',
    ],
  },
  earring: {
    title: '귀걸이',
    items: [
      '두 짝을 **포개지 않게** 옆으로 나란히 두고 촬영해 주세요. 겹치면 실루엣이 합쳐져 부피가 과대 추정될 수 있습니다.',
      '카드 **위에 올리지 말고** 카드 옆 바닥에 놓아 주세요.',
    ],
  },
  other: {
    title: '기타',
    items: [
      '귀금속과 신용카드를 **같은 바닥 면**에 두고, 카드 위에 올리지 마세요.',
      '형태가 복잡하면 가능한 한 **겹침·가림**을 줄여 주세요.',
    ],
  },
};

function ShootingTipLine({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <span>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

export function HomePage() {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const [knows, setKnows] = useState<'yes' | 'no' | null>(null);
  const [refWeight, setRefWeight] = useState('');
  const [files, setFiles] = useState<Partial<Record<(typeof VIEWS)[number], File>>>({});
  const [captureMode, setCaptureMode] = useState<CaptureMode>('camera');
  const [shotMode, setShotMode] = useState<ShotMode>('single');
  const [singleFile, setSingleFile] = useState<File | undefined>();
  const [metal, setMetal] = useState('gold');
  const [purity, setPurity] = useState('18k');
  const [product, setProduct] = useState('ring');
  const [thicknessMm, setThicknessMm] = useState('');
  const [declaredGoldG, setDeclaredGoldG] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [job, setJob] = useState<JobDto | null>(null);

  const onFile = (v: (typeof VIEWS)[number], f: File | undefined) => {
    setFiles((prev) => ({ ...prev, [v]: f }));
  };

  /** 금속을 바꾸면 함량 표기 체계가 달라지므로 첫 항목으로 재설정한다 (금 18k → 백금 pt950) */
  const onMetalChange = (next: string) => {
    setMetal(next);
    const first = PURITY_BY_METAL[next]?.[0]?.value;
    if (first) setPurity(first);
  };

  const poll = useCallback(async (id: string) => {
    for (let i = 0; i < 120; i++) {
      const j = await getJob(id);
      setJob(j);
      if (j.status === 'completed' || j.status === 'completed_low_confidence' || j.status === 'failed')
        return;
      await new Promise((r) => setTimeout(r, 1500));
    }
    setErr('시간 초과 — 나중에 job id로 다시 조회해 보세요.');
  }, []);

  const submit = async () => {
    setErr(null);
    setBusy(true);
    setJob(null);
    try {
      const fd = new FormData();
      if (shotMode === 'single') {
        if (!singleFile) throw new Error('사진을 준비해 주세요.');
        fd.append('image', singleFile);
      } else {
        for (const v of VIEWS) {
          if (!files[v]) throw new Error(`${VIEW_LABEL[v]} 이미지를 준비해 주세요.`);
        }
        for (const v of VIEWS) fd.append(v, files[v]!);
      }
      fd.append('capture_mode', shotMode);
      fd.append('metal', metal);
      fd.append('purity', purity);
      fd.append('product_k', product);
      if (isFlatProduct && !isPlated && thicknessMm.trim()) {
        fd.append('reference_thickness_mm', thicknessMm.trim());
      }
      if (isPlated && declaredGoldG.trim()) {
        fd.append('declared_gold_g', declaredGoldG.trim());
      }
      if (knows === 'yes' && refWeight.trim()) fd.append('reference_weight_g', refWeight.trim());
      if (knows) fd.append('knows_weight', knows);
      const { id } = await postJob(fd);
      await poll(id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  /** worker `constants.FLAT_PRODUCTS` — 두께가 깊이 노이즈보다 얇아 사진으로 못 재는 제품 */
  const isFlatProduct = product === 'goldbar' || product === 'plated';
  /** worker `constants.VOLUME_UNMEASURABLE_PRODUCTS` — 부피로 금 함량을 알 수 없는 제품 */
  const isPlated = product === 'plated';

  const CLIENT_ABSURD_MASS_G = 350;
  const tier = job?.result?.confidence_tier;
  const hideQuote = tier === 'low';
  const sanity = job?.result?.meta?.sanity;
  const fusion = job?.result?.meta?.scale_fusion;
  const recon = job?.result?.meta?.reconstruction;
  const quote = job?.quote ?? null;
  const won = (v: number) => v.toLocaleString('ko-KR');
  const massG = job?.result?.mass_est_g;
  const hasMass = typeof massG === 'number' && Number.isFinite(massG);
  const hideMass =
    !hasMass ||
    Boolean(sanity?.suppress_mass_display) ||
    (hasMass && massG > CLIENT_ABSURD_MASS_G);
  const workflow = job?.workflow ?? job?.result?.meta?.workflow ?? null;
  const retryViews =
    (job?.error?.retryViews && job.error.retryViews.length > 0
      ? job.error.retryViews
      : workflow?.retry_views) ?? [];
  const retrySet = new Set(retryViews);

  return (
    <div>
      {step === 0 && (
        <section className="card card--hero">
          <h2 className="section-title">귀금속 무게, 알고 계신가요?</h2>
          <p className="text-muted">
            저울 위에 올려 찍은 사진은 분석에 맞지 않을 수 있어요. 평평한 곳에 내려놓고 촬영해 주세요.
          </p>
          <div className="btn-row">
            <button type="button" className="btn btn--primary" onClick={() => { setKnows('no'); setStep(1); }}>
              아니요 / 잘 모르겠어요
            </button>
            <button type="button" className="btn btn--secondary" onClick={() => { setKnows('yes'); setStep(1); }}>
              네, 알고 있어요
            </button>
          </div>
        </section>
      )}

      {step === 1 && (
        <section className="card">
          <h2 className="section-title">촬영 전 체크</h2>
          <CaptureGuide />
          <ul className="check-list">
            <li>
              <strong>신용카드를 함께 찍어 주세요.</strong> 카드는 크기가 규격(85.6×54mm)으로
              고정돼 있어 사진 속 실제 크기를 재는 <strong>기준자</strong>가 됩니다.
              없으면 정확도가 크게 떨어집니다.
            </li>
            <li>신용카드와 귀금속을 <strong>같은 바닥 면</strong>에 둡니다.</li>
            <li>카드가 살짝 <strong>비스듬히</strong> 보이게 찍으면 거리 추정이 더 정확해집니다.</li>
            <li><strong>카드 위에 올리지 마세요.</strong> 카드 옆에 나란히 둡니다.</li>
            <li>저울 위에 올린 상태로 찍지 않습니다.</li>
            <li>정면·상단·좌·우·후면 순서로 촬영합니다.</li>
          </ul>
          <div className="notice notice--slate">
            <strong>텅 빈·속 빈 귀금속</strong>처럼 사진만으로 내부 구조를 알 수 없는 경우, 실제 금속 질량과 추정치 차이가 클 수 있습니다.{' '}
            Archimedes는 이 경우 <strong>정확한 질량을 제공하기 어렵습니다</strong>. 참고용 추정이며, 금은방 등에서 저울로 확정하시기 바랍니다.
          </div>
          {knows === 'yes' && (
            <div className="field" style={{ marginTop: '1rem' }}>
              <label htmlFor="refWeight">참고 무게 (g, 선택)</label>
              <input
                id="refWeight"
                type="number"
                step="0.01"
                value={refWeight}
                onChange={(e) => setRefWeight(e.target.value)}
              />
            </div>
          )}
          <div className="btn-row">
            <button type="button" className="btn btn--primary" onClick={() => setStep(2)}>다음</button>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className="card">
          <h2 className="section-title">각도별 촬영</h2>
          <p className="text-muted" style={{ marginTop: 0 }}>
            먼저 형태를 고른 뒤, 아래 <strong>형태별 촬영 주의</strong>를 확인하고 각도별 사진을 준비해 주세요.
          </p>

          <div className="form-grid" style={{ marginBottom: '1rem' }}>
            <div className="field">
              <label htmlFor="metal">금속</label>
              <select id="metal" value={metal} onChange={(e) => onMetalChange(e.target.value)}>
                {METALS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="purity">함량</label>
              <select id="purity" value={purity} onChange={(e) => setPurity(e.target.value)}>
                {(PURITY_BY_METAL[metal] ?? []).map((pp) => (
                  <option key={pp.value} value={pp.value}>{pp.label}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="product">형태</label>
              <select id="product" value={product} onChange={(e) => setProduct(e.target.value)}>
                <option value="ring">반지</option>
                <option value="necklace">목걸이</option>
                <option value="chain">체인</option>
                <option value="bracelet">팔찌</option>
                <option value="pendant">펜던트</option>
                <option value="earring">귀걸이</option>
                <option value="goldbar">골드바 · 골드카드 (순금)</option>
                <option value="plated">도금 · 금박 · 기념품</option>
                <option value="other">기타</option>
              </select>
            </div>
          </div>

          {isPlated ? (
            <div className="notice notice--tips" style={{ marginBottom: '1rem' }}>
              <p className="notice__title">이 제품은 무게를 계산하지 않습니다</p>
              <p className="text-small" style={{ marginTop: 0 }}>
                도금·금박 제품은 <strong>부피와 금 함량이 무관</strong>합니다. 사진으로는 잴 수
                없으니, 제품에 인쇄된 <strong>순금 함유량</strong>을 넣어 주시면 그 값으로
                시세 견적을 계산해 드립니다.
              </p>
              <div className="field" style={{ marginTop: '0.5rem' }}>
                <label htmlFor="declared">제품 표기 순금 함유량 (g)</label>
                <input
                  id="declared"
                  type="number"
                  step="0.001"
                  min="0.0001"
                  placeholder="예: 0.005"
                  value={declaredGoldG}
                  onChange={(e) => setDeclaredGoldG(e.target.value)}
                />
              </div>
              <p className="text-small" style={{ margin: '0.35rem 0 0' }}>
                비워 두면 무게·견적 없이 실측 치수만 보여 드립니다.
              </p>
            </div>
          ) : null}

          {isFlatProduct && !isPlated ? (
            <div className="notice notice--slate" style={{ marginBottom: '1rem' }}>
              <p className="notice__title">두께를 입력해 주세요</p>
              <p className="text-small" style={{ marginTop: 0 }}>
                골드바·골드카드는 두께가 <strong>0.3~0.5mm</strong> 수준이라
                사진으로는 잴 수 없습니다(측정 오차보다 얇음). 각인·제품 규격에 적힌 값을
                넣어 주시면 넓이는 사진에서 정확히 재서 무게를 계산합니다.
              </p>
              <div className="field" style={{ marginTop: '0.5rem' }}>
                <label htmlFor="thickness">두께 (mm)</label>
                <input
                  id="thickness"
                  type="number"
                  step="0.01"
                  min="0.05"
                  placeholder="예: 0.4"
                  value={thicknessMm}
                  onChange={(e) => setThicknessMm(e.target.value)}
                />
              </div>
              <p className="text-small" style={{ margin: '0.35rem 0 0' }}>
                모르면 비워 두셔도 됩니다. 대신 제품 기준값으로 가정해 오차가 커집니다.
              </p>
            </div>
          ) : null}

          {(() => {
            const tip = PRODUCT_SHOOTING_TIPS[product] ?? PRODUCT_SHOOTING_TIPS.other;
            return (
              <div className="notice notice--tips" style={{ marginBottom: '1rem' }}>
                <p className="notice__title">형태별 촬영 주의 — {tip.title}</p>
                <ul style={{ margin: 0, paddingLeft: '1.15rem' }}>
                  {tip.items.map((line, idx) => (
                    <li key={idx} style={{ marginBottom: '0.35rem' }}>
                      <ShootingTipLine text={line} />
                    </li>
                  ))}
                </ul>
              </div>
            );
          })()}

          <div className="btn-row" style={{ marginBottom: '0.5rem' }}>
            <button
              type="button"
              className={`btn btn--toggle ${shotMode === 'single' ? 'is-on' : ''}`}
              onClick={() => setShotMode('single')}
            >
              사진 1장
            </button>
            <button
              type="button"
              className={`btn btn--toggle ${shotMode === 'multiview' ? 'is-on' : ''}`}
              onClick={() => setShotMode('multiview')}
            >
              5방향 (고신뢰)
            </button>
          </div>

          <div className="btn-row" style={{ marginBottom: '0.5rem' }}>
            <button
              type="button"
              className={`btn btn--toggle ${captureMode === 'camera' ? 'is-on' : ''}`}
              onClick={() => setCaptureMode('camera')}
            >
              카메라로 촬영
            </button>
            <button
              type="button"
              className={`btn btn--toggle ${captureMode === 'upload' ? 'is-on' : ''}`}
              onClick={() => setCaptureMode('upload')}
            >
              파일에서 선택
            </button>
          </div>
          <p className="text-small" style={{ marginTop: 0 }}>
            {captureMode === 'camera'
              ? '휴대폰에서는 버튼을 누르면 카메라가 열립니다. 각 각도마다 1장씩 촬영하세요.'
              : '이미 찍어둔 사진이 있으면 각도별로 선택하세요.'}
          </p>

          {shotMode === 'single' ? <CaptureGuide compact /> : null}

          <div style={{ display: 'grid', gap: '0.75rem', marginBottom: '1rem', marginTop: '0.75rem' }}>
            {shotMode === 'single' ? (
              <label className={`file-slot ${retrySet.size > 0 ? 'file-slot--retry' : ''}`}>
                귀금속 + 신용카드가 함께 나온 사진 {singleFile ? `— ${singleFile.name}` : ''}
                <input
                  type="file"
                  accept="image/*"
                  capture={captureMode === 'camera' ? 'environment' : undefined}
                  onChange={(e) => setSingleFile(e.target.files?.[0])}
                  style={{ display: 'block', marginTop: '0.35rem' }}
                />
              </label>
            ) : (
              VIEWS.map((v) => (
                <label
                  key={v}
                  className={`file-slot ${retrySet.has(v) ? 'file-slot--retry' : ''}`}
                >
                  {VIEW_LABEL[v]} {files[v] ? `— ${files[v]!.name}` : ''}
                  {retrySet.has(v) ? ' (재촬영 권장)' : ''}
                  <input
                    type="file"
                    accept="image/*"
                    capture={captureMode === 'camera' ? 'environment' : undefined}
                    onChange={(e) => onFile(v, e.target.files?.[0])}
                    style={{ display: 'block', marginTop: '0.35rem' }}
                  />
                </label>
              ))
            )}
          </div>

          <button type="button" className="btn btn--primary" style={{ width: '100%' }} disabled={busy} onClick={() => void submit()}>
            {busy ? '처리 중…' : '분석 요청'}
          </button>
        </section>
      )}

      {err ? <p className="msg-error">{err}</p> : null}

      {job ? (
        <section className="card card--result" style={{ marginTop: '1rem' }}>
          <h3 className="section-title">결과</h3>
          <p>상태: <strong>{job.status}</strong></p>
          {job.error ? (
            <p className="msg-error" style={{ marginTop: '0.5rem' }}>
              {job.error.code}: {job.error.message}
              {job.error.errorSeverity === 'soft' ? ' (저신뢰 재시도 권장)' : ''}
            </p>
          ) : null}
          {retryViews.length > 0 ? (
            job?.result?.meta?.capture_mode === 'single' ? (
              <p className="msg-warn--soft">아래 안내를 반영해 사진을 다시 찍어 재요청해 주세요.</p>
            ) : (
              <p className="msg-warn--soft">
                다시 촬영은 전체가 아니라 <strong>{retryViews.map((v) => VIEW_LABEL[v as keyof typeof VIEW_LABEL] ?? v).join(', ')}</strong> 만 교체해서 재요청하세요.
              </p>
            )
          ) : null}
          {workflow?.degraded_reasons?.length ? (
            <p className="msg-warn--soft">저신뢰 사유: {workflow.degraded_reasons.join(', ')}</p>
          ) : null}
          {job.result ? (
            <>
              {hideMass ? (
                <p className="msg-warn">
                  {hasMass
                    ? '추정 무게가 비현실적으로 커 보여 숫자를 숨깁니다. 카드 옆 바닥에 놓고 다시 촬영해 주세요.'
                    : '이번 사진으로는 무게를 산출하지 못했습니다. 아래 안내를 보고 다시 시도해 주세요.'}
                </p>
              ) : (
                <p>
                  {sanity?.mass_source === 'declared_label' ? '표기 함유량: ' : '추정 무게: '}
                  <strong style={{ fontSize: '1.15rem', color: 'var(--gold-deep)' }}>{massG!.toFixed(3)} g</strong>
                </p>
              )}
              {job.result.confidence_tier ? (
                <p>신뢰도 등급: {job.result.confidence_tier} ({job.result.confidence_pct}%)</p>
              ) : null}
              {!hideMass && job.result.mass_range ? (
                <p>범위: {job.result.mass_range.min_g.toFixed(2)} ~ {job.result.mass_range.max_g.toFixed(2)} g</p>
              ) : null}
              <p className="text-small">
                {!hideMass ? `V_hull=${job.result.V_hull_mm3} mm³, V_adj=${job.result.V_adj_mm3} mm³, ` : ''}
                {job.result.algorithm_version ?? job.algorithmVersion ?? ''}
              </p>
              {fusion || recon ? (
                <div className="notice notice--slate" style={{ marginTop: '0.75rem' }}>
                  <p className="notice__title">측정값</p>
                  {fusion?.card_distance_mm != null ? (
                    <div>
                      카드까지 거리: {fusion.card_distance_mm.toFixed(0)} mm
                      {fusion.depth_rmse_mm != null ? ` (거리 오차 ±${fusion.depth_rmse_mm.toFixed(1)} mm)` : ''}
                    </div>
                  ) : null}
                  {recon?.length_mm != null && recon?.width_mm != null ? (
                    <div>실제 크기: {recon.length_mm.toFixed(1)} × {recon.width_mm.toFixed(1)} mm</div>
                  ) : null}
                  {recon?.h_mean_mm != null ? (
                    <div>
                      평균 두께: {recon.h_mean_mm.toFixed(1)} mm
                      {recon.thickness_clamp ? ' (관측 실패 — 기준값 가정)' : ''}
                    </div>
                  ) : null}
                  {fusion?.anchor_used === false ? (
                    <div>기준물(신용카드) 없이 추정한 값입니다.</div>
                  ) : null}
                </div>
              ) : null}
              {sanity?.warnings?.length ? (
                <ul className="msg-warn--soft" style={{ paddingLeft: '1.15rem' }}>
                  {sanity.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              ) : null}

              {quote && !isQuoteSuppressed(quote) ? (
                <div className="notice notice--tips" style={{ marginTop: '0.75rem' }}>
                  <p className="notice__title">예상 견적</p>
                  <p style={{ margin: '0.25rem 0' }}>
                    <strong style={{ fontSize: '1.2rem', color: 'var(--gold-deep)' }}>
                      {won(quote.estimate)} 원
                    </strong>
                    {quote.min != null && quote.max != null ? (
                      <span className="text-small"> ({won(quote.min)} ~ {won(quote.max)} 원)</span>
                    ) : null}
                  </p>
                  <p className="text-small" style={{ margin: 0 }}>
                    적용 시세 {won(quote.krwPerGram)} 원/g
                    {quote.buyRate < 1 ? ` · 매입률 ${(quote.buyRate * 100).toFixed(0)}%` : ''}
                    {' · '}
                    {quote.source}
                    {quote.stale ? ' (실시간 아님)' : ''}
                  </p>
                  <p className="text-small" style={{ margin: '0.35rem 0 0' }}>{quote.disclaimer}</p>
                </div>
              ) : (
                <p className="msg-warn--soft" style={{ marginTop: '0.75rem' }}>
                  {quote && isQuoteSuppressed(quote)
                    ? quote.message
                    : hideQuote
                      ? '신뢰도가 낮아 참고 시세·원화 견적은 표시하지 않습니다.'
                      : '시세를 가져오지 못해 금액을 표시하지 않습니다.'}
                </p>
              )}
            </>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
