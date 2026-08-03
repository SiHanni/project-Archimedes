import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * 렌더 중 예외가 나도 **흰 화면을 보이지 않게** 한다.
 *
 * 실제 사고: soft 에러 job 의 `result` 가 분석 결과가 아닌데 프런트가
 * `result.mass_est_g.toFixed(3)` 를 호출해 TypeError → React 가 트리를 통째로
 * 언마운트 → 사용자는 사진 5장을 찍고 빈 화면만 봤다.
 *
 * 근본 원인(서버 계약)은 따로 고쳤지만, 방어선이 없으면 같은 사고가 또 난다.
 */
type Props = { children: ReactNode };
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error('[Archimedes] render error', error, info.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <section className="card">
        <h2 className="section-title">화면을 그리는 중 문제가 생겼습니다</h2>
        <p className="text-muted">
          분석 요청 자체는 서버에 저장되었을 수 있습니다. 아래 내용을 개발자에게 알려 주세요.
        </p>
        <pre
          style={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: '0.8rem',
            background: '#f8fafc',
            padding: '0.6rem',
            borderRadius: 6,
          }}
        >
          {error.message}
        </pre>
        <div className="btn-row">
          <button type="button" className="btn btn--primary" onClick={this.reset}>
            다시 시도
          </button>
          <button
            type="button"
            className="btn btn--secondary"
            onClick={() => window.location.reload()}
          >
            새로고침
          </button>
        </div>
      </section>
    );
  }
}
