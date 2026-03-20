import { useCallback, useState } from 'react';
import { getJob, postJob, type JobDto } from '../api';

const VIEWS = ['front', 'top', 'left', 'right', 'back'] as const;
const VIEW_LABEL: Record<(typeof VIEWS)[number], string> = {
  front: '정면',
  top: '상단',
  left: '좌측',
  right: '우측',
  back: '후면',
};

type CaptureMode = 'camera' | 'upload';

export function HomePage() {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const [knows, setKnows] = useState<'yes' | 'no' | null>(null);
  const [refWeight, setRefWeight] = useState('');
  const [files, setFiles] = useState<Partial<Record<(typeof VIEWS)[number], File>>>({});
  const [captureMode, setCaptureMode] = useState<CaptureMode>('camera');
  const [metal, setMetal] = useState('gold');
  const [purity, setPurity] = useState('18k');
  const [product, setProduct] = useState('ring');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [job, setJob] = useState<JobDto | null>(null);

  const onFile = (v: (typeof VIEWS)[number], f: File | undefined) => {
    setFiles((prev) => ({ ...prev, [v]: f }));
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
      for (const v of VIEWS) {
        if (!files[v]) {
          throw new Error(`${VIEW_LABEL[v]} 이미지를 준비해 주세요.`);
        }
      }
      const fd = new FormData();
      for (const v of VIEWS) fd.append(v, files[v]!);
      fd.append('metal', metal);
      fd.append('purity', purity);
      fd.append('product_k', product);
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

  const CLIENT_ABSURD_MASS_G = 350;
  const tier = job?.result?.confidence_tier;
  const hideQuote = tier === 'low';
  const sanity = job?.result?.meta?.sanity;
  const massG = job?.result?.mass_est_g;
  const hideMass =
    Boolean(sanity?.suppress_mass_display) ||
    (typeof massG === 'number' && massG > CLIENT_ABSURD_MASS_G);
  const retryViews =
    (job?.error?.retryViews && job.error.retryViews.length > 0
      ? job.error.retryViews
      : job?.result?.meta?.workflow?.retry_views) ?? [];
  const retrySet = new Set(retryViews);

  return (
    <div>
      {step === 0 && (
        <section>
          <h2>귀금속의 무게를 알고 계신가요?</h2>
          <p style={{ color: '#64748b', fontSize: '0.92rem' }}>
            저울 위에 올려 찍은 사진은 분석에 맞지 않을 수 있어요. 평평한 곳에 내려놓고 촬영해 주세요. (§9.1)
          </p>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '1rem' }}>
            <button type="button" onClick={() => { setKnows('no'); setStep(1); }}>아니요 / 잘 모르겠어요</button>
            <button type="button" onClick={() => { setKnows('yes'); setStep(1); }}>네, 알고 있어요</button>
          </div>
        </section>
      )}

      {step === 1 && (
        <section>
          <h2>촬영 전 체크</h2>
          <ul style={{ color: '#334155' }}>
            <li>신용카드와 귀금속을 <strong>같은 바닥 면</strong>에 둡니다.</li>
            <li><strong>카드 위에 올리지 마세요.</strong> 카드 옆에 나란히 둡니다.</li>
            <li>저울 위에 올린 상태로 찍지 않습니다.</li>
            <li>정면·상단·좌·우·후면 순서로 촬영합니다.</li>
          </ul>
          {knows === 'yes' && (
            <label style={{ display: 'block', marginTop: '1rem' }}>
              참고 무게 (g, 선택)
              <input
                type="number"
                step="0.01"
                value={refWeight}
                onChange={(e) => setRefWeight(e.target.value)}
                style={{ display: 'block', marginTop: '0.35rem', width: '100%', maxWidth: 280 }}
              />
            </label>
          )}
          <button type="button" style={{ marginTop: '1rem' }} onClick={() => setStep(2)}>다음</button>
        </section>
      )}

      {step === 2 && (
        <section>
          <h2>각도별 촬영/선택</h2>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <button
              type="button"
              onClick={() => setCaptureMode('camera')}
              style={{ background: captureMode === 'camera' ? '#dbeafe' : undefined }}
            >
              카메라로 촬영
            </button>
            <button
              type="button"
              onClick={() => setCaptureMode('upload')}
              style={{ background: captureMode === 'upload' ? '#dbeafe' : undefined }}
            >
              파일에서 선택
            </button>
          </div>
          <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: 0 }}>
            {captureMode === 'camera'
              ? '휴대폰에서는 버튼을 누르면 카메라가 열립니다. 각 각도마다 1장씩 촬영하세요.'
              : '이미 찍어둔 사진이 있으면 각도별로 선택하세요.'}
          </p>

          <div style={{ display: 'grid', gap: '0.75rem', marginBottom: '1rem' }}>
            {VIEWS.map((v) => (
              <label
                key={v}
                style={{
                  display: 'block',
                  padding: '0.45rem',
                  borderRadius: 6,
                  border: retrySet.has(v) ? '1px solid #ef4444' : '1px solid #e5e7eb',
                  background: retrySet.has(v) ? '#fef2f2' : '#fff',
                }}
              >
                {VIEW_LABEL[v]} {files[v] ? `- ${files[v]!.name}` : ''}
                {retrySet.has(v) ? ' (재촬영 권장)' : ''}
                <input
                  type="file"
                  accept="image/*"
                  capture={captureMode === 'camera' ? 'environment' : undefined}
                  onChange={(e) => onFile(v, e.target.files?.[0])}
                  style={{ display: 'block', marginTop: '0.25rem' }}
                />
              </label>
            ))}
          </div>

          <div style={{ display: 'grid', gap: '0.5rem', maxWidth: 320, marginBottom: '1rem' }}>
            <label>
              금속
              <select value={metal} onChange={(e) => setMetal(e.target.value)} style={{ display: 'block', width: '100%' }}>
                <option value="gold">금</option>
                <option value="silver">은</option>
              </select>
            </label>
            <label>
              함량
              <select value={purity} onChange={(e) => setPurity(e.target.value)} style={{ display: 'block', width: '100%' }}>
                <option value="24k">24K</option>
                <option value="18k">18K</option>
                <option value="14k">14K</option>
                <option value="sterling">925</option>
              </select>
            </label>
            <label>
              형태
              <select value={product} onChange={(e) => setProduct(e.target.value)} style={{ display: 'block', width: '100%' }}>
                <option value="ring">반지</option>
                <option value="necklace">목걸이</option>
                <option value="chain">체인</option>
                <option value="bracelet">팔찌</option>
                <option value="pendant">펜던트</option>
                <option value="earring">귀걸이</option>
                <option value="other">기타</option>
              </select>
            </label>
          </div>
          <button type="button" disabled={busy} onClick={() => void submit()}>
            {busy ? '처리 중…' : '분석 요청'}
          </button>
        </section>
      )}

      {err && <p style={{ color: '#b91c1c', marginTop: '1rem', whiteSpace: 'pre-wrap' }}>{err}</p>}

      {job && (
        <section style={{ marginTop: '1.5rem', padding: '1rem', background: '#fff', borderRadius: 8 }}>
          <h3>결과</h3>
          <p>상태: <strong>{job.status}</strong></p>
          {job.error && (
            <p style={{ color: '#b91c1c' }}>
              {job.error.code}: {job.error.message}
              {job.error.errorSeverity === 'soft' ? ' (저신뢰 재시도 권장)' : ''}
            </p>
          )}
          {retryViews.length > 0 && (
            <p style={{ color: '#92400e' }}>
              다시 촬영은 전체가 아니라 <strong>{retryViews.map((v) => VIEW_LABEL[v as keyof typeof VIEW_LABEL] ?? v).join(', ')}</strong> 만 교체해서 재요청하세요.
            </p>
          )}
          {job.result && (
            <>
              {hideMass ? (
                <p style={{ color: '#b45309' }}>
                  추정 무게가 비현실적으로 커 보여 숫자를 숨깁니다. 카드 옆 바닥에 놓고 다시 촬영해 주세요.
                </p>
              ) : (
                <p>추정 무게: <strong>{job.result.mass_est_g.toFixed(3)} g</strong></p>
              )}
              <p>신뢰도 등급: {job.result.confidence_tier} ({job.result.confidence_pct}%)</p>
              {!hideMass && job.result.mass_range && (
                <p>범위: {job.result.mass_range.min_g.toFixed(2)} ~ {job.result.mass_range.max_g.toFixed(2)} g</p>
              )}
              <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
                {!hideMass ? `V_hull=${job.result.V_hull_mm3} mm³, V_adj=${job.result.V_adj_mm3} mm³, ` : ''}
                {job.result.algorithm_version}
              </p>
              {job.result.meta?.workflow?.degraded_reasons?.length ? (
                <p style={{ fontSize: '0.85rem', color: '#92400e' }}>
                  저신뢰 사유: {job.result.meta.workflow.degraded_reasons.join(', ')}
                </p>
              ) : null}
              {hideQuote ? (
                <p style={{ marginTop: '0.75rem', color: '#92400e' }}>신뢰도가 낮아 참고 시세·원화 견적은 표시하지 않습니다.</p>
              ) : (
                <p style={{ marginTop: '0.75rem', color: '#64748b' }}>시세 연동 전 — 금액 견적 UI는 추후 연결합니다.</p>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
}
