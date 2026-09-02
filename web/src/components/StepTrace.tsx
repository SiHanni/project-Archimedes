/**
 * 단계별 연산 과정 표시.
 *
 * 심사에서 "각 단계마다 어떤 계산을 했고 수치가 어떻게 변했는지"를 화면 그대로
 * 캡처해 제출하라는 요구가 있었다. 결과만 크게 띄우면 그 근거를 알 수 없으므로,
 * 파이프라인이 기록한 중간값을 그대로 표로 낸다.
 *
 * 수치는 워커가 실제 처리 중에 남긴 값이며, 화면에서 다시 계산하지 않는다.
 * 화면이 자체 계산을 하면 서버 결과와 어긋날 수 있고, 그러면 캡처가 근거가 되지 못한다.
 */
import type { JobDto } from '../api';

type Stage = { step: string; note: string; px: number; frac: number };
type DistStep = { step: string; value: string; note: string };

function seg(job: JobDto | null) {
  return (job?.result?.meta?.segmentation ?? null) as Record<string, unknown> | null;
}
function dist(job: JobDto | null) {
  return (job?.result?.meta?.distance ?? null) as Record<string, unknown> | null;
}

export function StepTrace({ job }: { job: JobDto | null }) {
  const s = seg(job);
  const d = dist(job);
  const stages = (s?.stages as Stage[] | undefined) ?? [];
  const steps = (d?.steps as DistStep[] | undefined) ?? [];
  if (!stages.length && !steps.length) return null;

  const px = s?.area_px as number | undefined;
  const total = s?.image_px as number | undefined;

  return (
    <div className="trace">
      {stages.length > 0 && (
        <>
          <h4 className="trace__h">외곽선 추출 — 단계별 연산</h4>
          <div className="table-wrap">
            <table className="batch-table trace__t">
              <thead>
                <tr>
                  <th>단계</th>
                  <th className="n">마스크(px)</th>
                  <th className="n">화면 대비</th>
                  <th className="n">변화</th>
                  <th>판정식 · 적용 결과</th>
                </tr>
              </thead>
              <tbody>
                {stages.map((v, i) => {
                  const prev = i > 0 ? stages[i - 1].px : null;
                  const diff = prev == null ? null : v.px - prev;
                  return (
                    <tr key={v.step}>
                      <td>{v.step}</td>
                      <td className="n">{v.px.toLocaleString()}</td>
                      <td className="n">{(v.frac * 100).toFixed(2)}%</td>
                      <td className={`n ${diff && diff < 0 ? 'neg' : ''}`}>
                        {diff == null ? '—' : diff === 0 ? '변화 없음' : `${diff > 0 ? '+' : ''}${diff.toLocaleString()}`}
                      </td>
                      <td className="dim">{v.note}</td>
                    </tr>
                  );
                })}
                {px != null && total != null && (
                  <tr className="sum">
                    <td colSpan={1}>최종</td>
                    <td className="n">{px.toLocaleString()}</td>
                    <td className="n">{((px / total) * 100).toFixed(2)}%</td>
                    <td className="n">—</td>
                    <td className="dim">
                      전체 {total.toLocaleString()} px 중 {px.toLocaleString()} px ·
                      외곽선 꼭짓점 {(s?.polygon_points as number) ?? 0}개 ·
                      물체 {(s?.shape_count as number) ?? 0} · 구멍 {(s?.hole_count as number) ?? 0}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {steps.length > 0 && (
        <>
          <h4 className="trace__h">거리 추정 — 단계별 연산</h4>
          <div className="table-wrap">
            <table className="batch-table trace__t">
              <thead>
                <tr>
                  <th>단계</th>
                  <th className="n">값</th>
                  <th>산출 근거</th>
                </tr>
              </thead>
              <tbody>
                {steps.map((v) => (
                  <tr key={v.step}>
                    <td>{v.step}</td>
                    <td className="n strong">{v.value}</td>
                    <td className="dim">{v.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small trace__f">
            핀홀 카메라에서 물체 쪽 삼각형(밑변 S · 높이 Z)과 센서 쪽 삼각형(밑변 p · 높이 f)이
            닮은꼴이므로 <code>p : f = S : Z</code> 가 성립한다. 이를 Z 에 대해 풀면
            <code>Z = f × S / p</code> 이다.
          </p>
        </>
      )}
    </div>
  );
}
