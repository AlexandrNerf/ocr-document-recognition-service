// Автоматический индекс для поиска по всем страницам
// Собирает данные из всех компонентов страниц

export interface SearchableItem {
  page: string;
  path: string;
  title: string;
  content: string;
  elementId?: string;
  type: 'heading' | 'paragraph' | 'list-item' | 'code';
}

// Автоматически собираем индекс из всех страниц
export const searchIndex: SearchableItem[] = [
  // Обзор
  {
    page: 'Обзор',
    path: '/overview',
    title: 'О проекте',
    content: 'OCR Document Recognition Service — это сервис для автоматического распознавания текста на изображениях и PDF-документах. Система использует современные технологии машинного обучения на базе PyTorch Lightning и Hydra для детекции и распознавания текста.',
    elementId: 'about',
    type: 'heading'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'PyTorch Lightning',
    content: 'Система использует PyTorch Lightning для обучения и работы с моделями машинного обучения',
    elementId: 'about',
    type: 'paragraph'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Hydra',
    content: 'Использование Hydra для управления конфигурациями и гибкой настройки пайплайна обработки',
    elementId: 'about',
    type: 'paragraph'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Основные возможности',
    content: 'Мультиязычное распознавание — поддержка русского, английского и казахского языков. Множество форматов — работа с изображениями (JPEG, JPG, PNG) и PDF-документами. REST API — удобный интерфейс для интеграции с вашими приложениями. Визуализация результатов — получение изображений с выделенными текстовыми блоками и bounding boxes. Метрики качества — конфиденсы для детекции и распознавания текста.',
    elementId: 'features',
    type: 'heading'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Архитектура',
    content: 'Система состоит из нескольких компонентов: Детектор — находит текстовые области на изображении. Распознаватель (CRNN) — извлекает текст из найденных областей. Препроцессор — подготовка изображений к обработке. Постпроцессор — обработка результатов распознавания.',
    elementId: 'architecture',
    type: 'heading'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Мультиязычное распознавание',
    content: 'Поддержка русского, английского и казахского языков',
    elementId: 'features',
    type: 'list-item'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Bounding boxes',
    content: 'Визуализация результатов с выделенными текстовыми блоками и bounding boxes',
    elementId: 'features',
    type: 'list-item'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Конфиденсы',
    content: 'Метрики качества — конфиденсы для детекции и распознавания текста',
    elementId: 'features',
    type: 'list-item'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'Детектор',
    content: 'Находит текстовые области на изображении',
    elementId: 'architecture',
    type: 'list-item'
  },
  {
    page: 'Обзор',
    path: '/overview',
    title: 'CRNN',
    content: 'Распознаватель (CRNN) — извлекает текст из найденных областей',
    elementId: 'architecture',
    type: 'list-item'
  },
  
  // Старт Backend
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'Установка',
    content: 'Для работы с сервисом требуется Python 3.10.16 и менеджер пакетов Poetry. Используйте conda для создания окружения и poetry для установки зависимостей.',
    elementId: 'installation',
    type: 'heading'
  },
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'Запуск сервиса',
    content: 'Перейдите в папку backend/ocr-document-recognition-service/core и запустите python app.py. Сервис будет доступен по адресу http://localhost:8000',
    elementId: 'run-service',
    type: 'heading'
  },
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'Python 3.10.16',
    content: 'Для работы с сервисом требуется Python 3.10.16',
    elementId: 'installation',
    type: 'paragraph'
  },
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'Poetry',
    content: 'Менеджер пакетов Poetry для установки зависимостей. Команды: pip install poetry==2.2.1, poetry install',
    elementId: 'installation',
    type: 'paragraph'
  },
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'Conda',
    content: 'Использование conda для создания виртуального окружения: conda create -n ocr-project python=3.10.16, conda activate ocr-project',
    elementId: 'installation',
    type: 'paragraph'
  },
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'app.py',
    content: 'Запуск сервиса через python app.py в папке backend/ocr-document-recognition-service/core',
    elementId: 'run-service',
    type: 'paragraph'
  },
  {
    page: 'Старт Backend',
    path: '/quickstart',
    title: 'localhost:8000',
    content: 'Сервис доступен по адресу http://localhost:8000 после запуска',
    elementId: 'run-service',
    type: 'paragraph'
  },
  
  // Тестирование OCR
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: 'Тестирование OCR',
    content: 'Загрузите файл для распознавания. Поддерживаемые форматы: JPEG, JPG, PNG, PDF. Система отправит файл на обработку через /process и будет проверять статус через /status/{task_id}.',
    elementId: 'ocr-test',
    type: 'heading'
  },
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: 'Форматы файлов',
    content: 'Поддерживаемые форматы: JPEG, JPG, PNG, PDF',
    elementId: 'ocr-test',
    type: 'paragraph'
  },
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: 'API процесс',
    content: 'Отправка файла на обработку через /process, получение task_id, проверка статуса через /status/{task_id}',
    elementId: 'ocr-test',
    type: 'paragraph'
  },
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: 'Загрузка файла',
    content: 'Загрузите файл для распознавания. Поддерживаемые форматы: JPEG, JPG, PNG, PDF (максимум 10MB)',
    elementId: 'ocr-test',
    type: 'paragraph'
  },
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: 'Task ID',
    content: 'После отправки файла получаете task_id для отслеживания статуса обработки',
    elementId: 'ocr-test',
    type: 'paragraph'
  },
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: 'Результат распознавания',
    content: 'После завершения обработки отображается HTML результат с распознанным текстом и визуализацией',
    elementId: 'ocr-test',
    type: 'paragraph'
  },
  {
    page: 'Тестирование OCR',
    path: '/test-api',
    title: '10MB',
    content: 'Максимальный размер загружаемого файла: 10MB',
    elementId: 'ocr-test',
    type: 'paragraph'
  },
  
  // API Документация
  {
    page: 'API Документация',
    path: '/api',
    title: 'API Документация',
    content: 'Swagger UI документация для всех эндпоинтов API. Включает POST /process для загрузки файлов, GET /status/{task_id} для проверки статуса и GET /result/{task_id} для получения результатов.',
    elementId: 'api-docs',
    type: 'heading'
  },
  {
    page: 'API Документация',
    path: '/api',
    title: 'POST /process',
    content: 'Эндпоинт для загрузки файла для обработки. Возвращает task_id',
    elementId: 'api-docs',
    type: 'paragraph'
  },
  {
    page: 'API Документация',
    path: '/api',
    title: 'GET /status/{task_id}',
    content: 'Проверка статуса обработки задачи. Возвращает статус: process, done или error',
    elementId: 'api-docs',
    type: 'paragraph'
  },
  {
    page: 'API Документация',
    path: '/api',
    title: 'GET /result/{task_id}',
    content: 'Получение результатов обработки задачи в формате HTML с визуализацией',
    elementId: 'api-docs',
    type: 'paragraph'
  },
  {
    page: 'API Документация',
    path: '/api',
    title: 'Swagger UI',
    content: 'Интерактивная документация API через Swagger UI. Позволяет тестировать эндпоинты прямо в браузере',
    elementId: 'api-docs',
    type: 'paragraph'
  },
  {
    page: 'API Документация',
    path: '/api',
    title: 'ReDoc',
    content: 'Альтернативная документация API в формате ReDoc с улучшенным отображением',
    elementId: 'api-docs',
    type: 'paragraph'
  },
  {
    page: 'API Документация',
    path: '/api',
    title: 'OpenAPI',
    content: 'OpenAPI JSON спецификация для интеграции с другими инструментами',
    elementId: 'api-docs',
    type: 'paragraph'
  },
  
  // Научные статьи
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'EAST: An Efficient and Accurate Scene Text Detector',
    content: 'Классическая статья о детекции текста в естественных сценах. Представляет архитектуру EAST, которая эффективно обнаруживает текст произвольной ориентации в изображениях.',
    elementId: 'east:-an-efficient-and-accurate-scene-text-detector',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'CRNN: An End-to-End Trainable Neural Network',
    content: 'Фундаментальная работа по CRNN (Convolutional Recurrent Neural Network) - архитектуре, которая широко используется для распознавания текста. Описывает комбинацию CNN и RNN для последовательного распознавания.',
    elementId: 'an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'TrOCR: Transformer-based OCR',
    content: 'Современный подход с использованием трансформеров для OCR. Показывает эффективность pre-trained моделей на задачах распознавания текста, особенно для многоязычных сценариев.',
    elementId: 'trocr:-transformer-based-optical-character-recognition-with-pre-trained-models',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'CRAFT: Character Region Awareness for Text Detection',
    content: 'Метод детекции текста, основанный на предсказании регионов символов. CRAFT показывает высокую точность на текстах произвольной формы и ориентации.',
    elementId: 'craft:-character-region-awareness-for-text-detection',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'Scene Text Detection and Recognition: Recent Advances in Deep Learning',
    content: 'Обзорная статья о современных подходах к детекции и распознаванию текста с использованием глубокого обучения. Содержит сравнение различных методов и архитектур.',
    elementId: 'scene-text-detection-and-recognition:-recent-advances-in-deep-learning',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'Mask TextSpotter',
    content: 'End-to-end подход для обнаружения и распознавания текста произвольной формы. Использует маски сегментации для точного определения границ текста.',
    elementId: 'mask-textspotter:-an-end-to-end-trainable-neural-network-for-spotting-text-with-arbitrary-shapes',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'What Is Wrong With Scene Text Recognition Model Comparisons?',
    content: 'Критический анализ существующих методов распознавания текста. Предлагает стандартизированные бенчмарки и сравнивает различные архитектуры на единых датасетах.',
    elementId: 'what-is-wrong-with-scene-text-recognition-model-comparisons?-dataset-and-model-analysis',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'PaddleOCR',
    content: 'Практическая реализация легковесной OCR системы. Включает детекцию и распознавание текста, оптимизированную для production использования. Поддерживает множество языков.',
    elementId: 'paddleocr:-a-practical-ultra-lightweight-ocr-system',
    type: 'heading'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'OCR оптическое распознавание символов',
    content: 'Оптическое распознавание символов (OCR) и распознавание текста в естественных сценах',
    elementId: undefined,
    type: 'paragraph'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'Papers With Code',
    content: 'Дополнительные ресурсы: Papers With Code OCR Leaderboard для отслеживания современных достижений в области OCR',
    elementId: undefined,
    type: 'paragraph'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'ChineseOCR',
    content: 'Open Source OCR проект ChineseOCR для изучения реализации OCR систем',
    elementId: undefined,
    type: 'paragraph'
  },
  {
    page: 'Научные статьи',
    path: '/research',
    title: 'DocTR',
    content: 'DocTR - Document Text Recognition библиотека для работы с документами',
    elementId: undefined,
    type: 'paragraph'
  }
];

