import React from 'react';
import { papers } from '../utils/ResearchPapers';

const ResearchPapers: React.FC = () => {
  return (
    <div className="content_wrapper">
      <h1>Научные статьи по теме проекта</h1>
      
      <div className="docs-section">
        <p>
          Ниже представлены ключевые научные статьи и исследования в области 
          <strong className="orange_color"> оптического распознавания символов (OCR)</strong> и 
          <strong className="orange_color"> распознавания текста в естественных сценах</strong>, 
          которые легли в основу современных подходов к решению задач детекции и распознавания текста.
        </p>

        <div className="papers-list">
          {papers.map((paper, index) => (
            <div key={index} className="paper-item">
              <h3 id={paper.title.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')}>{paper.title}</h3>
              <div className="paper-meta">
                <p><strong>Авторы:</strong> {paper.authors}</p>
                <p><strong>Год:</strong> {paper.year}</p>
              </div>
              <p className="paper-description">{paper.description}</p>
              <div className="button_container right">
                <a 
                  href={paper.link} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="custom_button"
                >
                  Читать статью
                </a>
              </div>
            </div>
          ))}
        </div>

        <div className="external-links">
        <h2>Дополнительные ресурсы</h2>
          <p>
            <a 
              className="external-links-a" 
              href="https://paperswithcode.com/task/optical-character-recognition" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              Papers With Code - OCR Leaderboard
            </a>
          </p>
          <p>
            <a 
              className="external-links-a" 
              href="https://github.com/chineseocr/chineseocr" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              ChineseOCR - Open Source OCR Project
            </a>
          </p>
          <p>
            <a 
              className="external-links-a" 
              href="https://github.com/mindee/doctr" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              DocTR - Document Text Recognition
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResearchPapers;

