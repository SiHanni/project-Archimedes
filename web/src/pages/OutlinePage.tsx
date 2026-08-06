/**
 * 에라토스테네스 — 기준물 없이 **누끼 + 거리 추정**을 하는 탭.
 *
 * 아르키메데스 탭과 일부러 다르게 만든 것:
 * - **무게·견적을 표시하지 않는다.** 거리는 "제품 종류의 일반적 크기" 가정에서
 *   나온 추정치라 ±10~30% 인데, 무게로 넘어가면 그 오차가 세제곱이 된다.
 *   없는 값을 0 이나 '-' 로 채우면 "재긴 쟀는데 0이네"로 읽히므로 자리를 두지 않는다.
 * - 가정한 크기를 **항상 함께 보여 준다.** 가정이 곧 오차이기 때문.
 * - 마스크·누끼·폴리곤을 내려받게 한다(계획서 Step 1 오토라벨링 산출물).
 */
import { useEffect, useRef, useState } from 'react';
import { assetUrl, getJob, postJob, type JobDto } from '../api';

const POLL_MS = 1500;
const POLL_LIMIT = 80;

export function OutlinePage() {
  const [productK, setProductK] = useState('ring');
  const [knownLongMm, setKnownLongMm] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [job, setJob] = useState<JobDto | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function choose(f: File | null) {
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
    setJob(null);
    setError(null);
  }

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setJob(null);
    try {
      const form = new FormData();
      form.append('image', file);
      form.append('capture_mode', 'outline');
      form.append('product_k', productK);
      if (knownLongMm.trim()) form.append('known_long_mm', knownLongMm.trim());
      const created = await postJob(form);
      let tries = 0;
      const poll = async () => {
        tries += 1;
        try {
          const j = await getJob(created.id);
          setJob(j);
          if (j.status === 'completed' || j.status === 'completed_low_confidence' || j.status === 'failed') {
            setBusy(false);
            return;
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
          setBusy(false);
          return;
        }
        if (tries >= POLL_LIMIT) {
          setError('시간이 너무 오래 걸립니다. 잠시 후 다시 시도해 주세요.');
          setBusy(false);
          return;
        }
        timer.current = window.setTimeout(poll, POLL_MS);
      };
      timer.current = window.setTimeout(poll, POLL_MS);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  const seg = (job?.result?.meta?.segmentation ?? null) as Record<string, unknown> | null;
  const polygon = (seg?.polygon_xy as number[][] | undefined) ?? [];
  const hasAssets = Array.isArray(seg?.assets) && (seg?.assets as string[]).includes('overlay.jpg');
  const done = job?.status === 'completed' || job?.status === 'completed_low_confidence';
  const dist = job?.result?.meta?.distance;
  const notes = job?.result?.meta?.sanity?.warnings ?? [];

  function downloadPolygon() {
    if (!job) return;
    const payload = {
      image: { width: seg?.image_width, height: seg?.image_height },
      polygon_xy: polygon,
      source: seg?.appearance_source,
      job_id: job.id,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `outline-${job.id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <main className="page">
      <section className="card">
        <h2>외곽선 추출 · 거리 추정</h2>
        <p className="muted">
          사진 한 장에서 귀금속만 오려 내고 카메라와의 거리를 추정합니다. 신용카드가 없어도 됩니다.
        </p>

        <div className="notice notice--tips" style={{ marginTop: '0.75rem' }}>
          <p className="notice__title">거리는 어떻게 추정하나요</p>
          <p style={{ margin: 0 }}>
            제품 종류의 <strong>일반적인 크기</strong>를 가정해 계산합니다. 그래서
            <strong> 대략적인 값</strong>입니다(실측 오차 ±10~30%). 실제 크기를 아신다면
            아래에 입력해 주세요 — 훨씬 정확해집니다. <strong>무게는 계산하지 않습니다</strong>{' '}
            — 크기 가정의 오차가 무게에서는 세제곱으로 커지기 때문입니다.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginTop: '1rem' }}>
          <label>
            제품 종류
            <select value={productK} onChange={(e) => setProductK(e.target.value)}>
              <option value="ring">반지</option>
              <option value="earring">귀걸이</option>
              <option value="necklace">목걸이</option>
              <option value="bracelet">팔찌</option>
              <option value="goldbar">골드바</option>
            </select>
          </label>
          <label>
            실제 긴 쪽 크기 (mm, 알면)
            <input
              type="number"
              min="1"
              step="0.1"
              placeholder="예: 20"
              value={knownLongMm}
              onChange={(e) => setKnownLongMm(e.target.value)}
            />
          </label>
        </div>

        <ul style={{ margin: '0.75rem 0 0', paddingLeft: '1.15rem' }}>
          <li><strong>무늬 없는 바닥</strong> 위에 귀금속 하나만 두세요.</li>
          <li>화면 <strong>가운데</strong>에 크게 나오도록 가까이 찍어 주세요.</li>
          <li>물체가 <strong>화면 안에 통째로</strong> 들어와야 합니다.</li>
        </ul>

        <div style={{ marginTop: '1rem' }}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => choose(e.target.files?.[0] ?? null)}
          />
        </div>

        {preview && (
          <img
            src={preview}
            alt="선택한 사진 미리보기"
            style={{ marginTop: '0.75rem', maxWidth: '100%', borderRadius: 8 }}
          />
        )}

        <button
          type="button"
          className="primary"
          disabled={!file || busy}
          onClick={submit}
          style={{ marginTop: '1rem' }}
        >
          {busy ? '분석 중…' : '외곽선 추출 · 거리 추정'}
        </button>

        {error && (
          <p className="notice notice--error" style={{ marginTop: '1rem', whiteSpace: 'pre-wrap' }}>
            {error}
          </p>
        )}

        {job?.status === 'failed' && (
          <p className="notice notice--error" style={{ marginTop: '1rem' }}>
            {job.error?.message ?? '추출에 실패했습니다.'}
          </p>
        )}

        {done && hasAssets && (
          <div style={{ marginTop: '1.25rem' }}>
            <h3>추출 결과</h3>
            <img
              src={assetUrl(job.id, 'overlay.jpg')}
              alt="귀금속 외곽선을 표시한 이미지"
              style={{ maxWidth: '100%', borderRadius: 8 }}
            />
            {dist?.object_mm != null ? (
              <div className="notice notice--slate" style={{ marginTop: '0.75rem' }}>
                <p className="notice__title">카메라 ↔ 귀금속 거리 (추정)</p>
                <div style={{ fontSize: '1.35rem', fontWeight: 700 }}>
                  약 {dist.object_mm.toFixed(0)} mm
                </div>
                {dist.range_mm ? (
                  <div className="muted">
                    범위 {dist.range_mm[0].toFixed(0)}~{dist.range_mm[1].toFixed(0)} mm
                    {dist.assumed_long_mm != null
                      ? ` · 가정한 크기 ${dist.assumed_long_mm.toFixed(0)}mm`
                      : ''}
                  </div>
                ) : null}
              </div>
            ) : null}
            {notes.length ? (
              <ul className="muted" style={{ marginTop: '0.5rem', paddingLeft: '1.15rem' }}>
                {notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            ) : null}
            <p className="muted" style={{ marginTop: '0.5rem' }}>
              외곽선 꼭짓점 {polygon.length}개 · 화면의{' '}
              {(((seg?.area_frac as number) ?? 0) * 100).toFixed(2)}%
            </p>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
              <a href={assetUrl(job.id, 'cutout.png')} target="_blank" rel="noreferrer">
                누끼 PNG 내려받기
              </a>
              <a href={assetUrl(job.id, 'mask.png')} target="_blank" rel="noreferrer">
                마스크 PNG 내려받기
              </a>
              <button type="button" className="linklike" onClick={downloadPolygon}>
                폴리곤 JSON 내려받기
              </button>
            </div>
          </div>
        )}

        {done && !hasAssets && (
          <p className="notice" style={{ marginTop: '1rem' }}>
            외곽선은 찾았지만 이미지 저장에 실패했습니다. 잠시 후 다시 시도해 주세요.
          </p>
        )}
      </section>
    </main>
  );
}
