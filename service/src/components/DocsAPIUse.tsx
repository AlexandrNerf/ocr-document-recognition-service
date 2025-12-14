import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const DocsAPIUse: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [resultHtml, setResultHtml] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [taskId, setTaskId] = useState<string>('');
  const resultRef = useRef<HTMLDivElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf'];
      if (!allowedTypes.includes(file.type)) {
        setError('Поддерживаются только файлы: JPEG, JPG, PNG, PDF');
        setSelectedFile(null);
        return;
      }
      
      setSelectedFile(file);
      setError('');
      setResultHtml('');
      setTaskId('');
    }
  };

  const handleProcessFile = async () => {
    if (!selectedFile) {
      setError('Пожалуйста, выберите файл');
      return;
    }

    setLoading(true);
    setError('');
    setResultHtml('');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('file_format', selectedFile.type === 'application/pdf' ? 'pdf' : 'image');

      // Отправляем файл на обработку
      const processResponse = await axios.post(`${API_BASE}/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      const newTaskId = processResponse.data.task_id;
      setTaskId(newTaskId);

      // Проверяем статус обработки
      const checkStatus = async () => {
        try {
          const statusResponse = await axios.get(`${API_BASE}/status/${newTaskId}`, {
            timeout: 30000,
          });

          const status = statusResponse.data.status;

          if (status === 'done') {
            setResultHtml(statusResponse.data.html || '');
            setLoading(false);
          } else if (status === 'error') {
            setError(statusResponse.data.error || 'Ошибка обработки');
            setLoading(false);
          } else {
            // Повторяем проверку через 1 секунду
            setTimeout(checkStatus, 1000);
          }
        } catch (err: any) {
          setError(err.message || 'Ошибка проверки статуса');
          setLoading(false);
        }
      };

      checkStatus();
    } catch (err: any) {
      let errorMessage = 'Ошибка при обработке файла';
      
      if (err.response) {
        if (err.response.status === 400) {
          errorMessage = 'Неподдерживаемый формат файла';
        } else if (err.response.status === 413) {
          errorMessage = 'Файл слишком большой';
        } else if (err.response.status === 500) {
          errorMessage = 'Ошибка сервера при обработке файла';
        } else {
          errorMessage = `Ошибка сервера: ${err.response.status}`;
        }
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = 'Превышено время ожидания';
      } else if (err.code === 'ERR_NETWORK') {
        errorMessage = 'Ошибка сети. Проверьте, что сервер запущен';
      }
      
      setError(errorMessage);
      setLoading(false);
      console.error('Error:', err);
    }
  };

  // Создаем iframe для отображения HTML с Plotly
  useEffect(() => {
    if (resultHtml && resultRef.current) {
      resultRef.current.innerHTML = '';
      
      // Модифицируем HTML для центрирования содержимого
      const parser = new DOMParser();
      const doc = parser.parseFromString(resultHtml, 'text/html');
      
      // Добавляем стили для центрирования body и plotly-div
      const style = doc.createElement('style');
      style.textContent = `
        body {
          display: flex;
          justify-content: center;
          align-items: center;
          margin: 0;
          padding: 0;
        }
        #plotly-div {
          width: 100% !important;
          height: 100% !important;
        }
      `;
      
      // Вставляем стили в head
      if (doc.head) {
        doc.head.appendChild(style);
      } else {
        // Если нет head, создаем его
        const head = doc.createElement('head');
        head.appendChild(style);
        doc.documentElement.insertBefore(head, doc.body);
      }
      
      const modifiedHtml = doc.documentElement.outerHTML;
      
      const blob = new Blob([modifiedHtml], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      const iframe = document.createElement('iframe');
      iframe.src = url;
      iframe.className = 'plotly-iframe';
      
      resultRef.current.appendChild(iframe);
      
      return () => {
        URL.revokeObjectURL(url);
        if (resultRef.current) {
          resultRef.current.innerHTML = '';
        }
      };
    }
  }, [resultHtml]);

  return (
    <div className="content_wrapper">
      <h1 id="ocr-test">Тестирование OCR</h1>
      
      <div className="image_inverter_container">
        <div className="upload_section">
          <h2>Загрузите файл для распознавания</h2>
          <p className="format_info">Поддерживаемые форматы: JPEG, JPG, PNG, PDF (максимум 10MB)</p>
          <input
            type="file"
            accept=".jpg,.jpeg,.png,.pdf,image/jpeg,image/jpg,image/png,application/pdf"
            onChange={handleFileChange}
            className="file_input"
            id="fileInput"
            disabled={loading || !!resultHtml}
            onClick={(e) => {
              if (loading || resultHtml) {
                e.preventDefault();
                e.stopPropagation();
              }
            }}
          />
          <label 
            htmlFor="fileInput" 
            className={`custom_file_input ${(loading || resultHtml) ? 'disabled' : ''}`}
            onClick={(e) => {
              if (loading || resultHtml) {
                e.preventDefault();
                e.stopPropagation();
              }
            }}
          >
            Выбрать файл
          </label>
          
          {selectedFile && (
            <div className="file_info">
              <p><strong>Выбранный файл:</strong> {selectedFile.name}</p>
              <p><strong>Размер:</strong> {(selectedFile.size / 1024).toFixed(2)} KB</p>
            </div>
          )}
          
          {taskId && (
            <div className="file_info">
              <p><strong>Task ID:</strong> <span className="orange_color">{taskId}</span></p>
            </div>
          )}
          
          {!resultHtml && (
            <div className="button_container right">
              <button 
                onClick={handleProcessFile}
                disabled={!selectedFile || loading}
                className="custom_button white_button"
              >
                {loading ? 'Обработка...' : 'Распознать текст'}
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="error_message">
            {error}
          </div>
        )}

        {resultHtml && (
          <div className="result_section">
            <h2>Результат распознавания</h2>
            <div 
              ref={resultRef}
              className="result_html"
            />
            <div className="button_container right">
              <button 
                onClick={() => {
                  setResultHtml('');
                  setTaskId('');
                  setSelectedFile(null);
                }}
                className="custom_button"
              >
                Загрузить новый файл
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocsAPIUse;
