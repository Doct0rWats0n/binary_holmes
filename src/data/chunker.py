import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class TextDataset(Dataset):
    def __init__(self, data, chunk_size=100, chunk_overlap=50):
        if isinstance(data, list):  # Если data это список json'ов, то конвертирует в строку
            data = ''.join(data)
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.char_to_index = {char: idx for idx, char in enumerate(set(data))}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

        # Создание чанков с перекрытием
        self.chunks = [data[i:i + chunk_size] for i in
                       range(0, len(data) - chunk_size + 1, chunk_size - chunk_overlap)]

    def __len__(self) -> int:
        """Длина списка чанков"""
        return len(self.chunks)

    def __getitem__(self, idx) -> torch.Tensor:
        """Получение тензора по индексу"""
        chunk = self.chunks[idx]
        return self.chunk_to_tensor(chunk)

    def chunk_to_tensor(self, chunk) -> torch.Tensor:
        """Превращение чанка в тензор"""
        chunk_indices = [self.char_to_index[char] for char in chunk]
        tensor = torch.tensor(chunk_indices, dtype=torch.long)
        return tensor

    def get_parameters(self) -> Dict:
        """Получение параметров (размер чанка и размер перекрытия)"""
        parameters = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
        return parameters

    def get_chunks(self) -> List:
        """Выдача списка чанков"""
        return self.chunks

    def get_tensors(self) -> List[torch.Tensor]:
        return [self.chunk_to_tensor(chunk) for chunk in self.chunks]
