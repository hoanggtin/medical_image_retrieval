
class Config:
    NUM_CLASSES = 4
    EMBED_DIM = 256  # tương thích với CNN output
    GEMP = 3
    PATCH_SIZE = 16  # không dùng cho CNN
    NUM_HEADS = 8    # không dùng cho CNN
    MLP_DIM = 512    # không dùng cho CNN
    DEPTH = 6        # không dùng cho CNN
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 25
    BATCH_SIZE = 32
    IMG_SIZE = 224



    DATA_DIR = 'D:\\Job\\image_retrieval\\data\\COVID_Dataset'
    #MODEL_SAVE_PATH = 'D:\\Job\\image_retrieval\\models\\covid_cnn_fpn_rmac.pth'
    MODEL_SAVE_PATH = r'D:\Job\image_retrieval\models\checkpoints\best_cnn_model.pth'