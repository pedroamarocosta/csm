import numpy as np


def gera_huffman(probs: dict) -> dict:
    if len(probs) == 1:
        return dict(zip(probs.keys(), np.array([[0]])))
    if len(probs) == 2:
        return dict(zip(probs.keys(), np.array([[0], [1]])))

    sorted_probs = np.array(sorted(probs.items(), key=lambda x: x[1]))
    syb1, syb2 = sorted_probs[0, 0], sorted_probs[1, 0]
    prob_syb1, prob_syb2 = probs.pop(syb1), probs.pop(syb2)
    probs[syb2 + syb1] = prob_syb1 + prob_syb2

    syb_cod = gera_huffman(probs)
    syb12_cod = syb_cod.pop(syb2 + syb1)
    syb_cod[syb1], syb_cod[syb2] = np.append(syb12_cod, 0), np.append(syb12_cod, 1)

    return syb_cod


def codifica(msg: np.ndarray, dic: dict, img_mode: bool) -> np.ndarray:
    return codifica_img_mode_on(msg, dic).astype('uint8') if img_mode else codifica_img_mode_off(msg, dic).astype('uint8')


def codifica_img_mode_on(img: np.ndarray, dic: dict) -> np.ndarray:
    return np.insert(codifica_img_mode_off(np.array([chr(byte) for byte in img.ravel()]), dic),
                     0, np.array([img.shape[0], img.shape[1]]))


def codifica_img_mode_off(msg: np.ndarray, dic: dict) -> np.ndarray:

    bitstream = np.concatenate([dic.get(syb) for syb in msg], axis=None)
    extra_bits = 0 if len(bitstream) % 8 == 0 else 8 - len(bitstream) % 8
    bitstream = np.append(bitstream, np.zeros(extra_bits)).astype('uint8')
    byte_construct = lambda x: x[0] << 7 | x[1] << 6 | x[2] << 5 | x[3] << 4 | x[4] << 3 | x[5] << 2 | x[6] << 1 | x[7]
    bytestream = np.insert(np.concatenate([byte_construct(byte) for byte in np.reshape(bitstream, (round(len(bitstream)/8), 8))],
                                          axis=None),
                           0, np.uint8(extra_bits))
    return bytestream


def escrever(msg, filename):
    msg.tofile('{}.dat'.format(filename))
