import faiss
import numpy as np

def build_faiss_index(features, d):
    """
    features: ndarray (num_samples, feature_dim)
    d: dimension of feature vector
    """
    index = faiss.IndexFlatL2(d)
    index.add(features.astype(np.float32))  # FAISS yêu cầu float32
    return index

def search_index(index, query_feature, k=5):
    """
    query_feature: ndarray (1, feature_dim)
    Trả về: (distances, indices)
    """
    D, I = index.search(query_feature.astype(np.float32), k)
    return D[0], I[0]
