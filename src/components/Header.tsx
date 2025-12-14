import { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Fuse from 'fuse.js';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import { searchIndex } from '../utils/searchIndex';

interface SearchResult {
  page: string;
  path: string;
  title: string;
  snippet: string;
  elementId?: string;
  score?: number;
}

interface HeaderProps {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  sidebarToggleRef?: React.RefObject<HTMLButtonElement | null>;
}

const Header: React.FC<HeaderProps> = ({ sidebarOpen, onToggleSidebar, sidebarToggleRef }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Настройка Fuse.js для более точного поиска с поддержкой небольших опечаток
  const fuse = useMemo(() => {
    return new Fuse(searchIndex, {
      keys: [
        { name: 'title', weight: 0.4 },
        { name: 'content', weight: 0.35 },
        { name: 'page', weight: 0.25 }
      ],
      threshold: 0.2, // Более строгий порог для точности (0.0 = точное совпадение)
      minMatchCharLength: 2,
      includeScore: true,
      includeMatches: true,
      ignoreLocation: false, // Учитываем позицию совпадения
      findAllMatches: true,
      shouldSort: true, // Сортировка по релевантности
    });
  }, []);

  // Функция для подсветки совпадений в тексте
  const highlightMatches = (text: string, matches: Array<{ indices: readonly [number, number][] }> | undefined): string => {
    if (!matches || matches.length === 0) {
      return text;
    }

    // Собираем все индексы совпадений
    const indices: number[][] = [];
    matches.forEach(match => {
      if (match.indices) {
        match.indices.forEach((range) => {
          const [start, end] = range;
          indices.push([start, end + 1]);
        });
      }
    });

    // Сортируем индексы по позиции
    indices.sort((a, b) => a[0] - b[0]);

    // Объединяем перекрывающиеся индексы
    const merged: number[][] = [];
    indices.forEach(([start, end]) => {
      if (merged.length === 0 || start > merged[merged.length - 1][1]) {
        merged.push([start, end]);
      } else {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
      }
    });

    // Создаем HTML с подсветкой
    let highlighted = '';
    let lastIndex = 0;
    merged.forEach((range) => {
      const [start, end] = range;
      highlighted += text.substring(lastIndex, start);
      highlighted += `<mark class="search-highlight">${text.substring(start, end)}</mark>`;
      lastIndex = end;
    });
    highlighted += text.substring(lastIndex);

    return highlighted;
  };

  // Поиск с использованием Fuse.js
  const performSearch = (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    const results = fuse.search(query, { limit: 10 });
    
    const formattedResults: SearchResult[] = results.map(result => {
      const item = result.item;
      const matches = result.matches || [];
      
      // Находим лучшее совпадение для создания сниппета
      let snippet = item.content;
      let snippetMatches: Array<{ indices: readonly [number, number][] }> = [];
      
      if (snippet.length > 150) {
        // Пытаемся найти позицию совпадения в content
        const contentMatch = matches.find(m => m.key === 'content');
        if (contentMatch && contentMatch.indices && contentMatch.indices.length > 0) {
          const matchIndex = contentMatch.indices[0][0];
          const start = Math.max(0, matchIndex - 50);
          const end = Math.min(item.content.length, matchIndex + 100);
          snippet = item.content.substring(start, end);
          
          // Корректируем индексы для нового сниппета
          snippetMatches = [{
            ...contentMatch,
            indices: contentMatch.indices.map((range) => {
              const [s, e] = range;
              return [s - start, e - start] as [number, number];
            }) as [number, number][]
          }];
          
          if (start > 0) snippet = '...' + snippet;
          if (end < item.content.length) snippet = snippet + '...';
        } else {
          snippet = snippet.substring(0, 150) + '...';
        }
      } else {
        // Используем совпадения из content для полного текста
        snippetMatches = matches.filter(m => m.key === 'content');
      }

      // Подсвечиваем совпадения в сниппете
      const highlightedSnippet = highlightMatches(snippet, snippetMatches.length > 0 ? snippetMatches : undefined);
      
      // Подсвечиваем совпадения в заголовке
      const titleMatches = matches.filter(m => m.key === 'title');
      const highlightedTitle = highlightMatches(item.title, titleMatches.length > 0 ? titleMatches : undefined);

      return {
        page: item.page,
        path: item.path,
        title: highlightedTitle, // Теперь это HTML с подсветкой
        snippet: highlightedSnippet, // Теперь это HTML с подсветкой
        elementId: item.elementId,
        score: result.score
      };
    });

    // Сортируем по релевантности (score - чем меньше, тем лучше)
    formattedResults.sort((a, b) => (a.score || 1) - (b.score || 1));

    setSearchResults(formattedResults);
    setShowResults(formattedResults.length > 0);
  };

  useEffect(() => {
    performSearch(searchQuery);
  }, [searchQuery]);

  // Закрытие результатов при клике вне области поиска
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Закрытие результатов при смене страницы
  useEffect(() => {
    setShowResults(false);
    setSearchQuery('');
  }, [location.pathname]);

  const handleResultClick = (result: SearchResult) => {
    navigate(result.path);
    setShowResults(false);
    setSearchQuery('');
    
    // Скроллим к элементу после навигации
    setTimeout(() => {
      if (result.elementId) {
        const element = document.getElementById(result.elementId);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      } else {
        // Если нет ID, скроллим к заголовку h2 с соответствующим текстом
        const headings = document.querySelectorAll('h2');
        headings.forEach(heading => {
          if (heading.textContent?.includes(result.title)) {
            heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      }
    }, 100);
  };

  const handleCloseSearch = () => {
    setShowResults(false);
    setSearchQuery('');
  };

  const shouldShowBackdrop = (showResults && searchResults.length > 0) || (searchQuery.trim().length > 0 && searchResults.length === 0);

  return (
    <>
      {shouldShowBackdrop && (
        <div className="sidebar-backdrop" onClick={handleCloseSearch} />
      )}
      <header className="app-header">
      <button
        ref={sidebarToggleRef}
        className="sidebar-toggle"
        onClick={onToggleSidebar}
        aria-label={sidebarOpen ? 'Скрыть навигацию' : 'Показать навигацию'}
      >
        {sidebarOpen ? <CloseIcon /> : <MenuIcon />}
      </button>

      <div className="header-search" ref={searchRef}>
        <div className="search-input-container">
          <SearchIcon className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Поиск по документации..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => searchQuery && setShowResults(searchResults.length > 0)}
          />
        </div>

        {showResults && searchResults.length > 0 && (
          <div className="search-results-container">
            <div className="search-results">
              {searchResults.map((result, index) => (
                <div
                  key={index}
                  className="search-result-item"
                  onClick={() => handleResultClick(result)}
                >
                  <h3 className="search-result-page">{result.page}</h3>
                  <p className="search-result-title" dangerouslySetInnerHTML={{ __html: result.title }} />
                  <div className="search-result-snippet" dangerouslySetInnerHTML={{ __html: result.snippet }} />
                </div>
              ))}
              
            </div>
          </div>
        )}

        {searchQuery.trim().length > 0 && searchResults.length === 0 && (
          <div className="search-results-container">
            <div className="search-results">
              <div className="search-no-results">Ничего не найдено</div>
            </div>
          </div>
        )}
      </div>
    </header>
    </>
  );
};

export default Header;

