/**
 * 외곽선 추출 — 누끼만. 기준물도 거리 추정도 없다.
 *
 * 거리 탭과 나눈 이유는 **비용**이다. 거리는 1GB 모델을 2~3분 돌리는데,
 * 라벨링 데이터셋을 만들려고 누끼만 뽑는 사용자에게 그 시간을 물릴 이유가 없다.
 * 이 탭은 몇 초면 끝난다.
 */
import { useJobUpload } from '../hooks/useJobUpload';
import { OutlineResult } from '../components/OutlineResult';

// 누끼만이면 몇 초면 끝난다. 넉넉히 2분.
const TIMEOUT_MS = 120_000;

export function OutlinePage() {
  const { file, preview, job, busy, error, done, choose, submit } = useJobUpload(TIMEOUT_MS);

  return (
    <main className="page">
      <section className="card">
        <h2>외곽선 추출</h2>
        <p className="muted">
          사진 한 장에서 귀금속만 오려 냅니다. 신용카드 같은 기준물이 없어도 됩니다.
        </p>

        <div className="notice notice--tips" style={{ marginTop: '0.75rem' }}>
          <p className="notice__title">이 탭이 하는 일</p>
          <p style={{ margin: 0 }}>
            외곽선·마스크·누끼 이미지와 <strong>폴리곤 좌표(JSON)</strong>를 만듭니다.
            학습용 라벨로 그대로 쓸 수 있습니다. <strong>크기·거리·무게는 계산하지
            않습니다</strong> — 거리는 <strong>거리 측정</strong> 탭, 무게는{' '}
            <strong>분석</strong> 탭입니다.
          </p>
        </div>

        <ul style={{ margin: '0.75rem 0 0', paddingLeft: '1.15rem' }}>
          <li><strong>무늬 없는 바닥</strong> 위에 귀금속 하나만 두세요.</li>
          <li>화면 <strong>가운데</strong>에 크게 나오도록 가까이 찍어 주세요.</li>
          <li>물체가 <strong>화면 안에 통째로</strong> 들어와야 합니다.</li>
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
          onClick={() => submit({ capture_mode: 'outline' })}
          style={{ marginTop: '1rem' }}
        >
          {busy ? '추출 중…' : '외곽선 추출'}
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
            <h3>추출 결과</h3>
            <OutlineResult job={job} />
          </div>
        )}
      </section>
    </main>
  );
}
