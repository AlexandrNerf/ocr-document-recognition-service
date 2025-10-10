from collections import Counter

import numpy as np
import torch
from doctr.models.recognition.crnn.pytorch import CTCPostProcessor

duplicate_map = {
    'А': 'A',
    'а': 'a',
    'В': 'B',
    'в': 'b',
    'Е': 'E',
    'е': 'e',
    'К': 'K',
    'к': 'k',
    'М': 'M',
    'м': 'm',
    'Н': 'H',
    'н': 'h',
    'О': 'O',
    'о': 'o',
    'Р': 'P',
    'р': 'p',
    'С': 'C',
    'с': 'c',
    'Т': 'T',
    'т': 't',
    'У': 'Y',
    'у': 'y',
    'Х': 'X',
    'х': 'x',
}

lang_defs = {
    "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "ru": "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "kz": "ӘІҢҒҮҰҚӨҺәіңғүұқөһАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя",
}

# Нормализуем символы
norm_chars_by_lang = {
    lang: set(duplicate_map.get(c, c) for c in chars)
    for lang, chars in lang_defs.items()
}

# Подсчёт в каких языках встречается символ
char_lang_count = Counter()
for chars in norm_chars_by_lang.values():
    for c in chars:
        char_lang_count[c] += 1

# Группировка (для полного вокаба)
# common = {c for c, count in char_lang_count.items() if count > 1}
en_only = set([c for c in lang_defs['en']])  # norm_chars_by_lang["en"] - common
ru_only = set([c for c in lang_defs['ru']])  # norm_chars_by_lang["ru"] - common
kz_only = set([c for c in lang_defs['kz']])  # norm_chars_by_lang["kz"] - common

summary = ''.join(sorted(en_only | ru_only | kz_only))

# Итоговый vocab с общими символами

VOCAB_MULTI: str = summary + '0123456789!"#$%&)*+,-./:;<=>?@[\]^_`{|}~«»“”’—–₸₽'

char_to_idx = {c: i for i, c in enumerate(VOCAB_MULTI)}

# Индексы по группам
GROUP_IDXS: dict[str, set[int]] = {
    "en": {char_to_idx[c] for c in en_only},
    "ru": {char_to_idx[c] for c in ru_only},
    "kz": {char_to_idx[c] for c in kz_only},
}


class MaskedCTCDecoder(CTCPostProcessor):
    def __init__(
        self,
        vocab: str = VOCAB_MULTI,
        group_indices: dict[str, set[int]] = GROUP_IDXS,
        detect_frames: int = 3,
    ):
        super().__init__(vocab)
        self.group_indices = group_indices
        self.detect_frames = detect_frames

    def detect_language(self, logits: torch.Tensor) -> str:
        # Используем только первые N срезов
        top_logits = logits[: self.detect_frames]
        top_indices = torch.argmax(top_logits, dim=-1)

        lang_counts = {lang: 0 for lang in ['en', 'ru', 'kz']}
        for idx in top_indices:
            for lang in ['en', 'ru', 'kz']:
                if idx.item() in self.group_indices[lang]:
                    lang_counts[lang] += 1

        return max(lang_counts, key=lang_counts.get)

    def call(self, logits: torch.Tensor):
        logits = logits.cpu()
        detected_lang = self.detect_language(logits)

        # Разрешённые индексы: common + detected_lang
        allowed = self.group_indices["common"] | self.group_indices[detected_lang]

        # Маскируем запрещённые
        mask = torch.full_like(logits, -1e9)
        mask[:, np.array(allowed)] = 0
        masked_logits = logits + mask

        return super().__call__(masked_logits)
