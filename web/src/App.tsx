import { Link, Route, Routes } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { LegalPage } from './pages/LegalPage';

export default function App() {
  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: '1.25rem' }}>
      <header style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: '1.35rem', margin: 0 }}>Archimedes</h1>
        <p style={{ margin: '0.35rem 0 0', color: '#64748b', fontSize: '0.9rem' }}>
          5방향 사진 업로드 → 참고 부피·무게 추정 (v1 스캐폴드)
        </p>
        <nav style={{ marginTop: '0.75rem', display: 'flex', gap: '1rem', fontSize: '0.9rem' }}>
          <Link to="/">분석</Link>
          <Link to="/legal/terms">이용약관</Link>
          <Link to="/legal/privacy">개인정보</Link>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/legal/terms" element={<LegalPage kind="terms" />} />
        <Route path="/legal/privacy" element={<LegalPage kind="privacy" />} />
      </Routes>
    </div>
  );
}
