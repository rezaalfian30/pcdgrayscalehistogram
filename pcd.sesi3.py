import numpy as np
import imageio
import matplotlib.pyplot as plt

image_path = 'gambar.RGB.jpg'     
image = imageio.imread(image_path)

if len(image.shape) == 3:
    
    R = image[..., 0].astype(np.float32)
    G = image[..., 1].astype(np.float32)
    B = image[..., 2].astype(np.float32)

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    gray = np.clip(gray, 0, 255).astype(np.uint8)
else:
    gray = image.astype(np.uint8)

print("Ukuran citra RGB       :", image.shape)
print("Ukuran citra grayscale :", gray.shape)
print("Total piksel           :", gray.size)

hist_gray, bins_gray = np.histogram(
    gray.flatten(),
    bins=256,
    range=[0, 256]
)

print("\nIntensitas | Jumlah Piksel")
print("----------------------------")
for i in range(256):
    print(f"{i:>9} | {hist_gray[i]:>10}")

print("\nTotal dari histogram :", hist_gray.sum())
print("Total piksel citra   :", gray.size)


plt.figure(figsize=(10, 6))
plt.bar(range(256), hist_gray, width=1.0, color='gray', edgecolor='black')
plt.title('Histogram Citra Grayscale')
plt.xlabel('Intensitas Piksel (0–255)')
plt.ylabel('Jumlah Piksel')
plt.xlim([0, 255])
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



dominan = int(np.argmax(hist_gray))
persen  = hist_gray[dominan] / gray.size * 100

print("\n>> Intensitas dominan :", dominan)
print(">> Jumlah piksel      :", hist_gray[dominan])
print(f">> Persentase         : {persen:.2f}% dari total piksel")
print("\n>> Intensitas dominan :", dominan)