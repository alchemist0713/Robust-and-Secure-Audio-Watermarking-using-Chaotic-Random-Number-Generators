import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import soundfile as sf
from PIL import Image

# ------------ Arnold's Cat Map ------------
def arnold_cat_map(img, iterations):
    n = img.shape[0]
    for _ in range(iterations):
        new_img = np.zeros_like(img)
        for x in range(n):
            for y in range(n):
                new_x = (x + y) % n
                new_y = (x + 2 * y) % n
                new_img[new_x, new_y] = img[x, y]
        img = new_img
    return img

def inverse_arnold_cat_map(img, iterations):
    n = img.shape[0]
    for _ in range(iterations):
        new_img = np.zeros_like(img)
        for x in range(n):
            for y in range(n):
                new_x = (2 * x - y) % n
                new_y = (-x + y) % n
                new_img[new_x, new_y] = img[x, y]
        img = new_img
    return img

# ------------ Chaotic Map: Logistic ------------
def logistic_map(length, x0=0.5, alpha=3.99):
    x = x0
    seq = []
    for _ in range(length):
        x = alpha * x * (1 - x)
        seq.append(x)
    return np.array(seq)

# ------------ Hamming Code ------------
def hamming_encode(bits):
    encoded = []
    for i in range(0, len(bits), 4):
        d = bits[i:i + 4]
        while len(d) < 4: d.append(0)
        p1 = d[0] ^ d[1] ^ d[3]
        p2 = d[0] ^ d[2] ^ d[3]
        p3 = d[1] ^ d[2] ^ d[3]
        encoded += [p1, p2] + d[:1] + [p3] + d[1:]
    return encoded

def hamming_decode(bits):
    decoded = []
    for i in range(0, len(bits), 7):
        if i + 7 > len(bits): break
        b = bits[i:i + 7]
        p1, p2, d1, p3, d2, d3, d4 = b
        c1 = p1 ^ d1 ^ d2 ^ d4
        c2 = p2 ^ d1 ^ d3 ^ d4
        c3 = p3 ^ d2 ^ d3 ^ d4
        error = c1 + c2 * 2 + c3 * 4
        if 1 <= error <= 7:
            b[error - 1] ^= 1
        decoded += [b[2], b[4], b[5], b[6]]
    return decoded

# ------------ Watermark Embedding ------------
def embed_watermark(audio, fs, watermark, alpha=0.01, iterations=5):
    f, t, Zxx = stft(audio, fs)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    flat_mag = mag.flatten()

    scrambled = arnold_cat_map(watermark, iterations)
    bits = [int(p > 128) for p in scrambled.flatten()]
    encoded_bits = hamming_encode(bits)
    chaos_seq = logistic_map(len(encoded_bits), x0=0.7)
    indices = (chaos_seq * len(flat_mag)).astype(int)

    for i, bit in enumerate(encoded_bits):
        flat_mag[indices[i]] += alpha if bit else -alpha

    mag_mod = flat_mag.reshape(mag.shape)
    Z_mod = mag_mod * np.exp(1j * phase)
    _, watermarked_audio = istft(Z_mod, fs)
    return watermarked_audio, encoded_bits, indices

# ------------ Watermark Extraction ------------
def extract_watermark(audio, fs, indices, bit_len, alpha=0.01, iterations=5, shape=(32, 32)):
    f, t, Zxx = stft(audio, fs)
    mag = np.abs(Zxx).flatten()
    extracted_bits = [1 if mag[i] > 0 else 0 for i in indices[:bit_len]]

    decoded = hamming_decode(extracted_bits)
    binary_img = np.array(decoded[:shape[0]*shape[1]]).reshape(shape)
    unscrambled = inverse_arnold_cat_map(binary_img * 255, iterations)
    return unscrambled.astype(np.uint8), decoded

# ------------ Evaluation Metrics ------------
def compute_metrics(original, extracted):
    original = np.array(original).flatten()
    extracted = np.array(extracted[:len(original)])
    errors = np.sum(original != extracted)
    ber = errors / len(original)
    acc = 100 * (1 - ber)
    return ber, acc

# ------------ MAIN ------------
if __name__ == "__main__":
    # File paths
    wav_path = "E:/req/filtered_audio.wav"
    image_path = "E:/req/sample_watermark.png"

    # Load audio and watermark
    audio, fs = sf.read(wav_path)
    image = Image.open(image_path).convert('L').resize((32, 32))
    wm_array = np.array(image)
    original_bits = [int(p > 128) for p in wm_array.flatten()]

    best_acc = 0
    best_img = None
    best_alpha = 0

    print("[INFO] Running accuracy test for 100+ alphas...")
    for alpha in np.linspace(0.001, 0.1, 150):
        try:
            watermarked_audio, encoded_bits, indices = embed_watermark(audio, fs, wm_array, alpha=alpha)
            extracted_img, extracted_bits = extract_watermark(watermarked_audio, fs, indices, len(encoded_bits), alpha=alpha)
            ber, acc = compute_metrics(original_bits, extracted_bits)
            print(f"Alpha = {alpha:.4f} | BER = {ber:.4f} | Accuracy = {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_img = extracted_img
                best_alpha = alpha
                sf.write("E:/req/final_watermarked.wav", watermarked_audio, fs)

        except Exception as e:
            print(f"[ERROR] Alpha {alpha:.4f} failed: {e}")

    print(f"\n[RESULT] Best Alpha = {best_alpha:.4f} with Accuracy = {best_acc:.2f}%")

    # Visual comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(wm_array, cmap='gray')
    axs[0].set_title("Original Watermark")
    axs[0].axis('off')

    axs[1].imshow(best_img, cmap='gray')
    axs[1].set_title(f"Extracted (\u03b1={best_alpha:.4f})")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
