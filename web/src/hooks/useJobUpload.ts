/**
 * 사진 한 장 올리고 job 이 끝날 때까지 기다리는 공통 훅.
 *
 * '외곽선' 탭과 '거리' 탭이 같은 업로드·폴링 로직을 쓴다. 두 페이지에 같은 코드를
 * 복붙하면 한쪽만 고쳐지는 사고가 난다(폴링 한도·타임아웃 같은 건 특히).
 *
 * ⚠️ 거리 모드는 Depth Pro(1GB) 를 돌려 **2~3분** 걸린다. 폴링 한도를 그 아래로
 * 잡으면 정상 job 을 "시간 초과"로 끊어 버린다.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { getJob, postJob, type JobDto } from '../api';

const POLL_MS = 2000;

export type UseJobUpload = {
  file: File | null;
  preview: string | null;
  job: JobDto | null;
  busy: boolean;
  error: string | null;
  done: boolean;
  choose: (f: File | null) => void;
  submit: (fields: Record<string, string>) => Promise<void>;
};

export function useJobUpload(timeoutMs: number): UseJobUpload {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [job, setJob] = useState<JobDto | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current);
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    },
    [],
  );

  const choose = useCallback((f: File | null) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const url = f ? URL.createObjectURL(f) : null;
    previewRef.current = url;
    setFile(f);
    setPreview(url);
    setJob(null);
    setError(null);
  }, []);

  const submit = useCallback(
    async (fields: Record<string, string>) => {
      if (!file) return;
      setBusy(true);
      setError(null);
      setJob(null);
      try {
        const form = new FormData();
        form.append('image', file);
        for (const [k, v] of Object.entries(fields)) {
          if (v !== '') form.append(k, v);
        }
        const created = await postJob(form);
        const limit = Math.ceil(timeoutMs / POLL_MS);
        let tries = 0;
        const poll = async () => {
          tries += 1;
          try {
            const j = await getJob(created.id);
            setJob(j);
            if (
              j.status === 'completed' ||
              j.status === 'completed_low_confidence' ||
              j.status === 'failed'
            ) {
              setBusy(false);
              return;
            }
          } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
            setBusy(false);
            return;
          }
          if (tries >= limit) {
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
    },
    [file, timeoutMs],
  );

  const done = job?.status === 'completed' || job?.status === 'completed_low_confidence';
  return { file, preview, job, busy, error, done, choose, submit };
}
