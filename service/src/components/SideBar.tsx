import { Link, useLocation } from 'react-router-dom';

type Props = {
  isOpen: boolean;
  onClose?: () => void;
};

const SideBar: React.FC<Props> = ({ isOpen, onClose }) => {
  const location = useLocation();

  const tabs = [
    { path: '/overview', label: 'Обзор' },
    { path: '/research', label: 'Научные статьи' },
    { path: '/quickstart', label: 'Старт Backend' },
    { path: '/test-api', label: 'Тестирование OCR' },
    { path: '/api', label: 'API Документация' },
  ];

  return (
    <div className={`sidebar-container ${isOpen ? 'open' : ''}`}>
      <nav className={`sidebar`}>
        <div className="sidebar-info">
          Навигация
        </div>

        {tabs.map((tab) => {
          const isActive = location.pathname === tab.path;
          return (
            <Link
              key={tab.path}
              to={tab.path}
              className={`sidebar_button ${isActive ? 'active' : ''}`}
              onClick={onClose}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
};

export default SideBar;
