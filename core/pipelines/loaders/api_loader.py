from shift_ocr.shift_ocr.loader import Loader
from shift_ocr.utils.base64utils import decode_image


class APILoader(Loader):

    def load(self, *args, **kwargs):
        if 'base64' in kwargs:
            image = decode_image(kwargs['base64'])
            return image
        else:
            raise ValueError('from_base64 is required')

    def end_stream(self):
        pass
