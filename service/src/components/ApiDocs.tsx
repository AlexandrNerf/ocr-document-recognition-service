import { useEffect, useRef } from "react";
import "swagger-ui-dist/swagger-ui.css";
import type SwaggerUIBundle from "swagger-ui-dist/swagger-ui-es-bundle";

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const ApiDocs: React.FC = () => {
  const uiRef = useRef<HTMLDivElement>(null);
  const swaggerUIRef = useRef<ReturnType<typeof SwaggerUIBundle> | null>(null);

  useEffect(() => {
    const loadSwaggerUI = async () => {
      if (!uiRef.current || swaggerUIRef.current) return;

      try {
        // Динамический импорт Swagger UI
        const SwaggerUIBundleModule = await import("swagger-ui-dist/swagger-ui-es-bundle");
        const SwaggerUIBundle = SwaggerUIBundleModule.default;
        
        // Очищаем контейнер перед инициализацией
        if (uiRef.current) {
          uiRef.current.innerHTML = '';
        }

        swaggerUIRef.current = SwaggerUIBundle({
          url: `${API_BASE}/openapi.json`,
          domNode: uiRef.current,
          docExpansion: "list",
          defaultModelsExpandDepth: 1,
          defaultModelExpandDepth: 1,
          displayRequestDuration: true,
          tryItOutEnabled: true,
          supportedSubmitMethods: ["get", "post", "put", "delete", "patch"],
          deepLinking: true,
          displayOperationId: false,
          filter: true,
          showExtensions: false,
          showCommonExtensions: false,
          syntaxHighlight: {
            activate: true,
            theme: "agate",
          },
          requestInterceptor: (request: any) => {
            return request;
          },
          responseInterceptor: (response: any) => {
            return response;
          },
          onComplete: () => {
            console.log("Swagger UI загружен");
          },
          onFailure: (error: any) => {
            console.error("Ошибка загрузки Swagger UI:", error);
            if (uiRef.current) {
              uiRef.current.innerHTML = `
                <div style="padding: 20px; text-align: center;">
                  <h3>Ошибка загрузки Swagger UI</h3>
                  <p>Не удалось загрузить документацию. Убедитесь, что сервер запущен на ${API_BASE}</p>
                  <p><a href="${API_BASE}/docs" target="_blank">Открыть Swagger UI в новой вкладке</a></p>
                </div>
              `;
            }
          },
        });
      } catch (error) {
        console.error("Ошибка при загрузке Swagger UI:", error);
        if (uiRef.current) {
          uiRef.current.innerHTML = `
            <div style="padding: 20px; text-align: center;">
              <h3>Ошибка загрузки Swagger UI</h3>
              <p>Не удалось загрузить документацию. Убедитесь, что сервер запущен на ${API_BASE}</p>
              <p><a href="${API_BASE}/docs" target="_blank">Открыть Swagger UI в новой вкладке</a></p>
            </div>
          `;
        }
      }
    };

    loadSwaggerUI();

    // Очистка при размонтировании
    return () => {
      if (uiRef.current) {
        uiRef.current.innerHTML = '';
      }
      swaggerUIRef.current = null;
    };
  }, []);

  return (
    <div className="content_wrapper">
      <h1>API Документация</h1>
      <div className="external-links">
        <h2>Внешние ссылки</h2>
        <p>
          <a className="external-links-a" href={`${API_BASE}/docs`} target="_blank" rel="noopener noreferrer">
            Swagger UI документация
          </a>
        </p>
        <p>
          <a className="external-links-a" href={`${API_BASE}/redoc`} target="_blank" rel="noopener noreferrer">
            ReDoc документация
          </a>
        </p>
        <p>
          <a className="external-links-a" href={`${API_BASE}/openapi.json`} target="_blank" rel="noopener noreferrer">
            OpenAPI JSON спецификация
          </a>
        </p>
      </div>
      <div className="api-docs-container">
        <div ref={uiRef} style={{ minHeight: "80vh", width: "100%" }} />
      </div>
    </div>
  );
};

export default ApiDocs;
