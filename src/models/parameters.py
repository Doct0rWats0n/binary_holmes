CHUNK_SIZE = 100
CHUNK_OVERLAP = 50

gru_params = {
    "EMBED_DIM": 64, "GRU_DIM": 64,
    "FC_DIM": 32, "FC_DROPOUT": 0.5,
    "LEAKY_COEF": 0.3, "GRU_NUM_LAYERS": 1,
}

lstm_params = {
    "EMBED_DIM": 128, "LSTM_DIM": 64,
    "FC_DIM": 32, "FC_DROPOUT": 0.5,
    "LEAKY_COEF": 0.3, "LSTM_NUM_LAYERS": 1
}