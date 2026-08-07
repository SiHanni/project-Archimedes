/**
 * 누끼 결과 표시 — '외곽선' 탭과 '거리' 탭이 함께 쓴다.
 *
 * 두 탭 모두 마스크를 만들므로 결과 화면의 이 부분은 같다. 복붙하면
 * 한쪽만 고쳐지므로 컴포넌트로 뺀다.
 */
import { assetUrl, type JobDto } from '../api';

type Props = {
  job: JobDto;
  /** 산출물 다운로드 링크를 보일지. 거리 탭에서는 군더더기라 끈다. */
  showDownloads?: boolean;
};

export function OutlineResult({ job, showDownloads = true }: Props) {
  const seg = (job.result?.meta?.segmentation ?? null) as Record<string, unknown> | null;
  const polygon = (seg?.polygon_xy as number[][] | undefined) ?? [];
  const hasAssets = Array.isArray(seg?.assets) && (seg?.assets as string[]).includes('overlay.jpg');
  if (!hasAssets) return null;

  function downloadPolygon() {
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
    <>
      <img
        src={assetUrl(job.id, 'overlay.jpg')}
        alt="귀금속 외곽선을 표시한 이미지"
        style={{ maxWidth: '100%', borderRadius: 8 }}
      />
      <p className="muted" style={{ marginTop: '0.5rem' }}>
        외곽선 꼭짓점 {polygon.length}개 · 화면의{' '}
        {(((seg?.area_frac as number) ?? 0) * 100).toFixed(2)}%
      </p>
      {showDownloads && (
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
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
      )}
    </>
  );
}
