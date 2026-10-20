import numpy as np


def l2_normalize(vector: list[float]) -> list[float]:
    """
    对向量做L2归一化
    :param vector: 原始向量（list 格式）
    :return: 归一化后的向量
    """
    if not vector:  # 空向量直接返回
        return vector

    # 提取非零维度的数值
    values = np.array(vector, dtype=np.float64)
    # 计算 L2 范数（避免除以 0）
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # 范数接近 0 时，直接返回原向量（避免除零错误）
        return vector

    # 归一化：每个数值除以 L2 范数
    normalized_values = values / l2_norm
    return normalized_values.tolist()