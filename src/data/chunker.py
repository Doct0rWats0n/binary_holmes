import torch
from typing import Dict
from src.models.parameters import CHUNK_SIZE, CHUNK_OVERLAP


class Chunker:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def __call__(self, data):
        """Вызов экземпляра класса с данными для их обработки"""
        return self.chunk_process(data)

    def change_size(self, chunk_size):
        """Изменение размера чанков"""
        self.chunk_size = chunk_size

    def change_overlap(self, chunk_overlap):
        """Изменение перекрытия"""
        self.chunk_overlap = chunk_overlap

    def chunk_to_tensor(self, chunks):
        """Преобразование чанков в тензор"""
        char_to_index = {char: idx for idx, char in enumerate(set(chunks))}
        chunk_indices = [char_to_index[char] for char in chunks]
        tensor = torch.tensor(chunk_indices, dtype=torch.long)
        return tensor

    def chunk_process(self, data):
        """Преобразование данных сначала в чанки, а потом в матрицу тензоров"""
        if isinstance(data, list):  # Если data это список json'ов, то конвертирует в строку
            data = ''.join(data)
        data_len = len(data)
        overlap_value = self.chunk_size - self.chunk_overlap
        chunks = [data[i:i + self.chunk_size] for i in
                  range(0, data_len - self.chunk_size + 1, overlap_value)]
        left_size = data_len % self.chunk_size
        while left_size > 0:  # Добавление паддингов

            padding = chunks[-1][overlap_value:]
            quotient, remainder = divmod(left_size, overlap_value)
            if quotient == 1 and remainder == 0:
                add_value = overlap_value
                padding_add = data[-add_value:]
            elif quotient > 0:
                add_value = overlap_value
                padding_add = data[-left_size: -(left_size - add_value)]
            else:
                add_value = remainder
                padding_add = data[-add_value:]
                padding_add += ' ' * (self.chunk_size - add_value - overlap_value)
            padding += padding_add
            chunks.append(padding)
            left_size -= add_value
        result = [self.chunk_to_tensor(chunk) for chunk in chunks]
        matrix = torch.stack(result)
        return matrix

    def get_parameters(self) -> Dict:
        """Получение параметров (размер чанка и размер перекрытия)"""
        parameters = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        return parameters
