import { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './styles.css';
import Header from './components/Header';
import SideBar from './components/SideBar';
import DocsOverview from './components/DocsOverview';
import DocsQuickStart from './components/DocsQuickStart';
import DocsAPIUse from './components/DocsAPIUse';
import ApiDocs from './components/ApiDocs';
import ResearchPapers from './components/ResearchPapers';
import NotFoundPage from './components/NotFoundPage';

function TrackedRoutes() {
  return (
    <Routes>
      <Route path="/overview" element={<DocsOverview />} />
      <Route path="/research" element={<ResearchPapers />} />
      <Route path="/quickstart" element={<DocsQuickStart />} />
      <Route path="/test-api" element={<DocsAPIUse />} />
      <Route path="/api" element={<ApiDocs />} />
      <Route path="/" element={<Navigate to="/overview" replace />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}

function AppContent() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const sidebarToggleRef = useRef<HTMLButtonElement>(null);

  const toggleSidebar = () => setSidebarOpen((prev) => !prev);
  const closeSidebar = () => setSidebarOpen(false);

  // Закрытие сайдбара при клике вне его области
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!sidebarOpen) return;

      const target = event.target as Node;
      
      if (
        sidebarRef.current &&
        !sidebarRef.current.contains(target) &&
        sidebarToggleRef.current &&
        !sidebarToggleRef.current.contains(target)
      ) {
        closeSidebar();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [sidebarOpen]);

  return (
    <header className="main-container">
      <Header 
        sidebarOpen={sidebarOpen} 
        onToggleSidebar={toggleSidebar}
        sidebarToggleRef={sidebarToggleRef}
      />
      <main className="main-panel">
        <div ref={sidebarRef}>
          <SideBar isOpen={sidebarOpen} onClose={closeSidebar} />
        </div>
        {sidebarOpen && <div className="sidebar-backdrop" onClick={closeSidebar} />}
        <div className="content">
          <TrackedRoutes />
        </div>
      </main>
    </header>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
