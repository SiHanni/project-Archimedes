/**
 * 거리 측정 — 기준물 없이 카메라↔귀금속 거리를 추정한다.
 *
 * 원리는 `worker/app/pipeline/distance.py` 머리말 참고. 요약하면
 * `거리 = 초점거리 × 실제크기 / 픽셀크기` 이고, 초점거리는 EXIF 또는 모델 추정,
 * 실제 크기는 제품 종류의 사전값(또는 사용자 입력)이다.
 *
 * 화면에서 지키는 것 두 가지:
 * - **점 하나가 아니라 범위**로 보여 준다. 가정에서 나온 값이라 점으로 쓰면
 *   실제보다 정밀해 보인다.
 * - **가정한 크기를 항상 함께** 보여 준다. 그게 오차의 대부분이기 때문.
 */
import { useState } from 'react';
import { useJobUpload } from '../hooks/useJobUpload';
import { OutlineResult } from '../components/OutlineResult';

// Depth Pro(1GB) 를 CPU 로 돌려 2~3분 걸린다. 여유를 두고 8분.
const TIMEOUT_MS = 480_000;

const PRODUCTS: Array<{ value: string; label: string; hint: string }> = [
  { value: 'ring', label: '반지', hint: '가장 정확합니다 (손가락 크기라 편차가 작음)' },
  { value: 'earring', label: '귀걸이', hint: '형태 편차가 커서 오차가 큽니다' },
  { value: 'necklace', label: '목걸이', hint: '뭉친 정도에 따라 오차가 큽니다' },
  { value: 'bracelet', label: '팔찌', hint: '' },
  { value: 'goldbar', label: '골드바', hint: '0.05g~100g 폭이 넓어 오차가 큽니다' },
];

export function DistancePage() {
  const [productK, setProductK] = useState('ring');
  const [knownLongMm, setKnownLongMm] = useState('');
  const { file, preview, job, busy, error, done, choose, submit } = useJobUpload(TIMEOUT_MS);

  const dist = job?.result?.meta?.distance;
  const notes = job?.result?.meta?.sanity?.warnings ?? [];
  const hint = PRODUCTS.find((p) => p.value === productK)?.hint ?? '';

  return (
    <main className="page">
      <section className="card">
        <h2>거리 측정</h2>
        <p className="muted">
          사진 한 장으로 카메라와 귀금속 사이 거리를 추정합니다. 신용카드가 없어도 됩니다.
        </p>

        <div className="notice notice--tips" style={{ marginTop: '0.75rem' }}>
          <p className="notice__title">어떻게 재나요 · 얼마나 맞나요</p>
          <p style={{ margin: 0 }}>
            제품 종류의 <strong>일반적인 크기</strong>를 가정해 계산합니다. 그래서 결과를
            점이 아니라 <strong>범위</strong>로 드립니다. 실측 오차는 반지 약 ±14%,
            골드바 약 ±31%였습니다. <strong>실제 크기를 아신다면 입력해 주세요</strong> —
            오차가 절반 이하로 줄어듭니다.
          </p>
          <p style={{ margin: '0.5rem 0 0' }}>
            📷 <strong>직접 찍은 원본 사진일수록 정확합니다.</strong> 캡처본·다운로드본·
            잘라낸 사진은 촬영 정보(EXIF)가 지워져 있어 <strong>렌즈 정보를 추정으로
            대신</strong> 하게 되고, 그만큼 오차 범위가 넓어집니다. 메신저로 주고받은
            사진도 대부분 여기 해당합니다.
          </p>
          <p style={{ margin: '0.5rem 0 0' }}>
            ⏱ 한 장에 <strong>약 30초</strong> 걸립니다. 외곽선만 필요하면{' '}
            <strong>외곽선 추출</strong> 탭이 더 빠릅니다.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginTop: '1rem' }}>
          <label>
            제품 종류
            <select value={productK} onChange={(e) => setProductK(e.target.value)}>
              {PRODUCTS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
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
        {hint && !knownLongMm.trim() && (
          <p className="muted" style={{ margin: '0.35rem 0 0' }}>
            {hint}
          </p>
        )}

        <ul style={{ margin: '0.75rem 0 0', paddingLeft: '1.15rem' }}>
          <li><strong>무늬 없는 바닥</strong> 위에 귀금속 하나만 두세요.</li>
          <li>화면 <strong>가운데</strong>에 크게, 통째로 나오게 찍어 주세요.</li>
          <li>
            폰 <strong>기본 카메라</strong>로 찍어 그대로 올려 주세요. 메신저를 거치면
            렌즈 정보가 지워져 추정에 시간이 더 걸립니다.
          </li>
        </ul>

        <div style={{ marginTop: '1rem' }}>
          <input type="file" accept="image/*" onChange={(e) => choose(e.target.files?.[0] ?? null)} />
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
          onClick={() =>
            submit({
              capture_mode: 'distance',
              product_k: productK,
              known_long_mm: knownLongMm.trim(),
            })
          }
          style={{ marginTop: '1rem' }}
        >
          {busy ? '측정 중… (2~3분)' : '거리 측정'}
        </button>

        {error && (
          <p className="notice notice--error" style={{ marginTop: '1rem', whiteSpace: 'pre-wrap' }}>
            {error}
          </p>
        )}

        {/*
          soft 에러(`completed_low_confidence`)는 status 가 'failed' 가 아니다.
          'failed' 만 보면 **에러 메시지를 아무 데도 안 띄우게 된다** — 실측:
          누끼 실패인데 화면엔 "아래 안내를 확인해 주세요"만 뜨고 안내가 없었다.
        */}
        {job?.error?.message && (
          <p className="notice notice--error" style={{ marginTop: '1rem' }}>
            {job.error.message}
          </p>
        )}

        {done && job && (
          <div style={{ marginTop: '1.25rem' }}>
            <h3>측정 결과</h3>
            {dist?.object_mm != null ? (
              <div className="notice notice--slate">
                <p className="notice__title">카메라 ↔ 귀금속 거리 (추정)</p>
                <div style={{ fontSize: '1.6rem', fontWeight: 700 }}>
                  약 {(dist.object_mm / 10).toFixed(1)} cm
                </div>
                {dist.range_mm && (
                  <div className="muted">
                    범위 {(dist.range_mm[0] / 10).toFixed(1)}~
                    {(dist.range_mm[1] / 10).toFixed(1)} cm
                    {dist.assumed_long_mm != null &&
                      ` · 가정한 크기 ${(dist.assumed_long_mm / 10).toFixed(1)}cm`}
                    {dist.size_source === 'user_input' && ' (직접 입력)'}
                  </div>
                )}
              </div>
            ) : job.error?.message ? null : (
              <p className="notice notice--error">
                거리를 계산하지 못했습니다.
              </p>
            )}
            {notes.length > 0 && (
              <ul className="muted" style={{ marginTop: '0.5rem', paddingLeft: '1.15rem' }}>
                {notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            )}
            <div style={{ marginTop: '1rem' }}>
              <OutlineResult job={job} showDownloads={false} />
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
