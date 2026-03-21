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
      <Link to="/" className="legal-back">
        ← 돌아가기
      </Link>
      <article className="card">
        {data ? (
          <>
            <h2 className="section-title">{data.title}</h2>
            <p className="legal-body">{data.body}</p>
          </>
        ) : (
          <p className="text-muted">로딩…</p>
        )}
      </article>
    </div>
  );
}
