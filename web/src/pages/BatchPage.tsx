/**
 * 여러 장을 한 번에 넣고 **표 하나로** 결과를 보는 화면 (심사 시연용).
 *
 * 한 장씩 올려 결과를 눈으로 옮겨 적는 방식은 10장이면 열 번 반복해야 하고,
 * 그 사이 값을 잘못 옮길 여지가 생긴다. 여기서는 전부 큐에 넣고 끝나는 대로
 * 같은 표에 채운다.
 *
 * ⚠️ 워커는 **한 번에 한 장**만 처리한다. 10장을 넣으면 순서대로 줄을 서므로
 *    총 시간은 (장수 × 장당 시간)이다. 실측 — 외곽선 약 25초/장, 거리 약 32초/장.
 *    그래서 화면에 '대기 중 / 처리 중 / 완료'를 구분해 보여 준다. 진행이 멈춘 게
 *    아니라 줄을 서 있다는 것을 알 수 있어야 한다.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { getJob, postJob, type JobDto } from '../api';

const POLL_MS = 2500;
// 거리 모드는 Depth Pro 를 돌려 한 장에 30초 안팎. 10장이 줄을 서면 6분을 넘길 수
// 있으므로 넉넉히 잡는다. 짧게 잡으면 정상 job 을 '시간 초과'로 끊는다.
const TIMEOUT_MS = 20 * 60 * 1000;

type Mode = 'outline' | 'distance';

type Row = {
  id: string;
  file: File;
  name: string;
  preview: string;
  productK: string;
  jobId: string | null;
  status: 'idle' | 'queued' | 'running' | 'done' | 'error';
  job: JobDto | null;
  error: string | null;
  startedAt: number | null;
  endedAt: number | null;
};

const PRODUCTS = [
  { k: 'ring', label: '반지' },
  { k: 'necklace', label: '목걸이' },
  { k: 'earring', label: '귀걸이' },
  { k: 'goldbar', label: '골드바' },
  { k: 'bracelet', label: '팔찌' },
  { k: 'other', label: '기타' },
];

/** mm → cm 문자열. 소수 한 자리까지만 — 그 아래는 이 방법의 정밀도를 넘는다. */
function cm(mm: number | null | undefined): string {
  if (mm == null || !Number.isFinite(mm)) return '—';
  return (mm / 10).toFixed(1);
}

function seg(job: JobDto | null) {
  return (job?.result?.meta?.segmentation ?? null) as Record<string, unknown> | null;
}
function dist(job: JobDto | null) {
  return (job?.result?.meta?.distance ?? null) as Record<string, number | string> | null;
}

export default function BatchPage() {
  const [mode, setMode] = useState<Mode>('outline');
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const abort = useRef(false);
  const urls = useRef<string[]>([]);

  useEffect(
    () => () => {
      abort.current = true;
      urls.current.forEach((u) => URL.revokeObjectURL(u));
    },
    [],
  );

  const addFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const next: Row[] = Array.from(files).map((f, i) => {
      const url = URL.createObjectURL(f);
      urls.current.push(url);
      return {
        id: `${Date.now()}-${i}-${f.name}`,
        file: f,
        name: f.name,
        preview: url,
        productK: 'ring',
        jobId: null,
        status: 'idle',
        job: null,
        error: null,
        startedAt: null,
        endedAt: null,
      };
    });
    setRows((prev) => [...prev, ...next]);
  }, []);

  const setRow = useCallback((id: string, patch: Partial<Row>) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }, []);

  const runAll = useCallback(async () => {
    if (!rows.length || running) return;
    abort.current = false;
    setRunning(true);

    // 전부 한꺼번에 제출한다. 워커가 하나씩 꺼내 가므로 큐 자체가 순서를 지킨다.
    const submitted: { id: string; jobId: string }[] = [];
    for (const r of rows) {
      if (abort.current) break;
      if (r.status === 'done') continue;
      try {
        const form = new FormData();
        form.append('capture_mode', mode);
        form.append('product_k', r.productK);
        form.append('metal', 'gold');
        form.append('purity', '24k');
        form.append('image', r.file);
        const res = await postJob(form);
        submitted.push({ id: r.id, jobId: res.id });
        setRow(r.id, { jobId: res.id, status: 'queued', startedAt: Date.now(), error: null });
      } catch (e) {
        setRow(r.id, { status: 'error', error: e instanceof Error ? e.message : String(e) });
      }
    }

    // 끝날 때까지 함께 폴링한다
    const deadline = Date.now() + TIMEOUT_MS;
    const pending = new Map(submitted.map((s) => [s.jobId, s.id]));
    while (pending.size && Date.now() < deadline && !abort.current) {
      await new Promise((res) => setTimeout(res, POLL_MS));
      for (const [jobId, rowId] of Array.from(pending.entries())) {
        try {
          const job = await getJob(jobId);
          if (job.status === 'completed' || job.status === 'completed_low_confidence') {
            setRow(rowId, { status: 'done', job, endedAt: Date.now() });
            pending.delete(jobId);
          } else if (job.status === 'failed') {
            setRow(rowId, {
              status: 'error',
              job,
              endedAt: Date.now(),
              error: job.error?.message ?? '실패',
            });
            pending.delete(jobId);
          } else {
            setRow(rowId, { status: job.status === 'processing' ? 'running' : 'queued' });
          }
        } catch {
          // 일시적인 네트워크 오류 — 다음 폴링에서 다시 본다
        }
      }
    }
    setRunning(false);
  }, [rows, mode, running, setRow]);

  const doneCount = rows.filter((r) => r.status === 'done').length;
  const errCount = rows.filter((r) => r.status === 'error').length;
  const busyCount = rows.filter((r) => r.status === 'queued' || r.status === 'running').length;

  return (
    <div className="batch">
      <h2>여러 장 한 번에</h2>
      <p className="muted">
        사진을 여러 장 고르면 차례로 처리해 아래 표에 정리합니다. 워커가 한 번에 한 장씩
        처리하므로 <b>10장이면 5~6분</b> 정도 걸립니다.
      </p>

      <div className="batch-controls">
        <label className="seg">
          <input
            type="radio"
            name="mode"
            checked={mode === 'outline'}
            onChange={() => setMode('outline')}
            disabled={running}
          />
          외곽선 추출 <span className="muted">(약 25초/장)</span>
        </label>
        <label className="seg">
          <input
            type="radio"
            name="mode"
            checked={mode === 'distance'}
            onChange={() => setMode('distance')}
            disabled={running}
          />
          거리 추정 <span className="muted">(약 32초/장)</span>
        </label>
      </div>

      <div className="batch-controls">
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={(e) => addFiles(e.target.files)}
          disabled={running}
        />
        <button type="button" onClick={runAll} disabled={!rows.length || running}>
          {running ? `처리 중… (${doneCount + errCount}/${rows.length})` : `${rows.length}장 분석`}
        </button>
        {rows.length > 0 && !running ? (
          <button
            type="button"
            className="ghost"
            onClick={() => {
              urls.current.forEach((u) => URL.revokeObjectURL(u));
              urls.current = [];
              setRows([]);
            }}
          >
            비우기
          </button>
        ) : null}
      </div>

      {rows.length > 0 ? (
        <p className="muted">
          완료 {doneCount} · 실패 {errCount} · 대기/처리 중 {busyCount} / 전체 {rows.length}
        </p>
      ) : null}

      {rows.length > 0 ? (
        <div className="table-wrap">
          <table className="batch-table">
            <thead>
              {mode === 'outline' ? (
                <tr>
                  <th>사진</th>
                  <th>파일</th>
                  <th className="n">마스크</th>
                  <th className="n">물체</th>
                  <th className="n">구멍</th>
                  <th className="n">걸린 시간</th>
                  <th>상태</th>
                </tr>
              ) : (
                <tr>
                  <th>사진</th>
                  <th>파일</th>
                  <th>종류</th>
                  <th className="n">거리</th>
                  <th className="n">범위</th>
                  <th className="n">오차</th>
                  <th className="n">걸린 시간</th>
                  <th>상태</th>
                </tr>
              )}
            </thead>
            <tbody>
              {rows.map((r) => {
                const s = seg(r.job);
                const d = dist(r.job);
                const took =
                  r.startedAt && r.endedAt ? `${Math.round((r.endedAt - r.startedAt) / 1000)}초` : '—';
                const statusLabel =
                  r.status === 'done'
                    ? '완료'
                    : r.status === 'error'
                      ? '실패'
                      : r.status === 'running'
                        ? '처리 중'
                        : r.status === 'queued'
                          ? '대기 중'
                          : '—';
                return (
                  <tr key={r.id} className={r.status === 'error' ? 'row-err' : undefined}>
                    <td>
                      <img className="thumb" src={r.preview} alt="" />
                    </td>
                    <td className="fname">{r.name}</td>
                    {mode === 'outline' ? (
                      <>
                        <td className="n">
                          {s?.area_frac != null ? `${(Number(s.area_frac) * 100).toFixed(2)}%` : '—'}
                        </td>
                        <td className="n">{(s?.shape_count as number) ?? '—'}</td>
                        <td className="n">{(s?.hole_count as number) ?? '—'}</td>
                      </>
                    ) : (
                      <>
                        <td>
                          <select
                            value={r.productK}
                            disabled={running}
                            onChange={(e) => setRow(r.id, { productK: e.target.value })}
                          >
                            {PRODUCTS.map((p) => (
                              <option key={p.k} value={p.k}>
                                {p.label}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td className="n strong">
                          {d?.object_mm != null ? `${cm(Number(d.object_mm))} cm` : '—'}
                        </td>
                        <td className="n">
                          {Array.isArray(d?.range_mm)
                            ? `${cm((d!.range_mm as unknown as number[])[0])}~${cm(
                                (d!.range_mm as unknown as number[])[1],
                              )} cm`
                            : '—'}
                        </td>
                        <td className="n">
                          {d?.relative_sigma != null
                            ? `±${Math.round(Number(d.relative_sigma) * 100)}%`
                            : '—'}
                        </td>
                      </>
                    )}
                    <td className="n">{took}</td>
                    <td>
                      <span className={`badge badge-${r.status}`}>{statusLabel}</span>
                      {r.error ? <div className="err-msg">{r.error}</div> : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      {mode === 'distance' ? (
        <div className="notice">
          <p>
            <b>직접 찍은 원본 사진일수록 정확합니다.</b> 캡처본·다운로드본·잘라낸
            사진은 촬영 정보(EXIF)가 지워져 있어 <b>초점거리를 추정으로 대신</b> 하게
            되고, 그만큼 오차 범위가 넓어집니다. 카카오톡·메신저로 주고받은 사진도
            대부분 이 경우에 해당합니다.
          </p>
          <p>
            거리는 <b>제품 종류의 대표 크기를 가정</b>해 계산합니다. 표의 ‘종류’를 실제와
            맞게 바꾸면 정확해집니다. 오차 폭이 종류마다 다른 것은 그 가정의 폭이 다르기
            때문입니다 — 반지는 손가락에 맞아야 해 좁고(±14%), 목걸이·귀걸이는 형태가
            자유로워 넓습니다(±41~46%).
          </p>
        </div>
      ) : null}
    </div>
  );
}
