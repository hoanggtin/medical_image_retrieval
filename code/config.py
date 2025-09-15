
class Config:
    IMG_SIZE = 224
    PATCH_SIZE = 16
    EMBED_DIM = 768
    NUM_CLASSES = 4
    DEPTH = 8
    NUM_HEADS = 8
    MLP_DIM = 2048
    GEMP = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4

    DATA_DIR = 'D:\\Job\\image_retrieval\\data\\COVID_Dataset'
    MODEL_SAVE_PATH = 'D:\\Job\\image_retrieval\\models\\covid_vit_fpn_rmac.pth'
    
