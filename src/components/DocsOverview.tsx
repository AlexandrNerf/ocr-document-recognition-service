import React from 'react';

const DocsOverview: React.FC = () => {
  return (
    <div className="content_wrapper">
      <h1>Обзор проекта</h1>
      
      <div className="docs-section">
        <h2 className="hover-hash">О проекте</h2>
        <p>
          <strong className="orange_color">OCR Document Recognition Service</strong> — это сервис для 
          автоматического распознавания текста на изображениях и PDF-документах. Система использует 
          современные технологии машинного обучения на базе <strong>PyTorch Lightning</strong> и 
          <strong> Hydra</strong> для детекции и распознавания текста.
        </p>

        <h2 id="features" className="hover-hash">Основные возможности</h2>
        <ul className="custom-list">
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Мультиязычное распознавание</strong> — поддержка русского, 
              английского и казахского языков</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Множество форматов</strong> — работа с изображениями 
              (JPEG, JPG, PNG) и PDF-документами</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">REST API</strong> — удобный интерфейс для интеграции 
              с вашими приложениями</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Визуализация результатов</strong> — получение 
              изображений с выделенными текстовыми блоками и bounding boxes</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Метрики качества</strong> — конфиденсы для детекции 
              и распознавания текста</span>
            </div>
          </li>
        </ul>

        <h2 id="architecture" className="hover-hash">Архитектура</h2>
        <p>
          Система состоит из нескольких компонентов:
        </p>
        <ul className="custom-list">
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Детектор</strong> — находит текстовые области на изображении</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Распознаватель (CRNN)</strong> — извлекает текст из найденных областей</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Препроцессор</strong> — подготовка изображений к обработке</span>
            </div>
          </li>
          <li>
            <div className="custom-list-text">
              <span><strong className="orange_color">Постпроцессор</strong> — обработка результатов распознавания</span>
            </div>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default DocsOverview;

