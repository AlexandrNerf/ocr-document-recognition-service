import React from 'react';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requireAdmin?: boolean;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children, requireAdmin = false }) => {

  if (requireAdmin) {
    return (
      <div className="error-container">
        <h2>Доступ запрещен</h2>
        <p>У вас недостаточно прав для доступа к этой странице.</p>
      </div>
    );
  }

  return <>{children}</>;
};

export default ProtectedRoute;
