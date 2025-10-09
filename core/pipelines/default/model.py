import base64

from pydantic import BaseModel, field_validator


class ImageBase64(BaseModel):
    data: str

    @field_validator('data')
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value)
            return value
        except Exception as e:
            raise ValueError(f"Некорректная base64-строка: {e}")
