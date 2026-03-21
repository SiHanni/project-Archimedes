import { Link, Route, Routes } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { LegalPage } from './pages/LegalPage';

export default function App() {
  return (
    <div className="layout">
      <header className="app-header">
        <div className="app-header__top">
          <div>
            <h1 className="app-brand">
              Archimedes
              <span className="app-brand__accent" aria-hidden />
            </h1>
            <p className="app-tagline">
              5방향 사진만으로 귀금속 <strong>참고 부피·무게</strong>를 추정합니다. 금은방 방문 전, 집에서 가볍게 확인해 보세요.
            </p>
          </div>
        </div>
        <nav className="app-nav" aria-label="주요 메뉴">
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
