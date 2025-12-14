declare module 'swagger-ui-dist/swagger-ui-es-bundle' {
  interface SwaggerUIOptions {
    url?: string;
    domNode?: HTMLElement | null;
    dom_id?: string;
    docExpansion?: string;
    defaultModelsExpandDepth?: number;
    defaultModelExpandDepth?: number;
    displayRequestDuration?: boolean;
    tryItOutEnabled?: boolean;
    supportedSubmitMethods?: string[];
    deepLinking?: boolean;
    displayOperationId?: boolean;
    filter?: boolean;
    showExtensions?: boolean;
    showCommonExtensions?: boolean;
    syntaxHighlight?: {
      activate?: boolean;
      theme?: string;
    };
    requestInterceptor?: (request: any) => any;
    responseInterceptor?: (response: any) => any;
    onComplete?: () => void;
    onFailure?: (error: any) => void;
    [key: string]: any;
  }

  const SwaggerUIBundle: (options: SwaggerUIOptions) => any;
  
  export default SwaggerUIBundle;
}

