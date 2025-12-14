import React from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const DocsQuickStart: React.FC = () => {

  return (
    <div className="content_wrapper">
      <h1>Быстрый старт backend</h1>
      
      <div className="docs-section">
        <h2 id="installation" className="hover-hash">Установка</h2>
        <p>Для работы с сервисом требуется <strong>Python 3.10.16</strong> и менеджер пакетов <strong>Poetry</strong>.</p>
        
        <div className="code-block">
          <code>{`conda create -n ocr-project python=3.10.16`}</code>
          <code>{`conda activate ocr-project`}</code>
          <code>{`pip install poetry==2.2.1`}</code>
          <code>{`poetry install`}</code>
        </div>

        <h2 id="run-service" className="hover-hash">Запуск сервиса</h2>
        <p>Перейдите в папку <code>backend/ocr-document-recognition-service/core</code> и запустите:</p>
        
        <div className="code-block">
          <code>{`python app.py`}</code>
        </div>

        <p>Сервис будет доступен по адресу <code>{API_BASE}</code></p>
      </div>
    </div>
  );
};

export default DocsQuickStart;

