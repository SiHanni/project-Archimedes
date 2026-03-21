import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiBase } from '../api';

export function LegalPage({ kind }: { kind: 'terms' | 'privacy' }) {
  const [data, setData] = useState<{ title: string; body: string } | null>(null);
  useEffect(() => {
    const path = kind === 'terms' ? 'legal/terms' : 'legal/privacy';
    fetch(`${apiBase}/${path}`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ title: '오류', body: '문서를 불러오지 못했습니다.' }));
  }, [kind]);
  return (
    <div>
      <Link to="/">← 돌아가기</Link>
      {data ? (
        <>
          <h2>{data.title}</h2>
          <p style={{ whiteSpace: 'pre-wrap' }}>{data.body}</p>
        </>
      ) : (
        <p>로딩…</p>
      )}
    </div>
  );
}
