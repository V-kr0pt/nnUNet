import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def save_central_slice(input_file, mask_file, output_file="overlay.png"):
    # carregar imagem e máscara
    image = nib.load(input_file).get_fdata()
    mask = nib.load(mask_file).get_fdata()

    # normalizar imagem para 0-1 (melhor contraste)
    image = image.astype(np.float32)
    image = image / np.max(image)

    # escolher slice central
    z = image.shape[2] // 2
    slice_image = image[:, :, z]
    slice_mask = mask[:, :, -1]
    slice_image = np.flip(np.rot90(slice_image), axis=1)
    slice_mask = np.flip(np.rot90(slice_mask), axis=1)
    
    # plotar e salvar
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_image, cmap="gray")
    plt.imshow(slice_mask, cmap="jet", alpha=0.3)  # sobreposição
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Salvo em {output_file}")

if __name__ == "__main__":
    # Exemplo de uso
    #image_names = ['4537110_PROC_L_MLO_20130105193311.nii.gz',
    #              '76201127_PROC_L_MLO_20150105143403.nii.gz',
    #              '91493466_PROC_L_MLO_20190527195023.nii.gz',
    #              '92463763_PROC_L_MLO_20200302182630.nii.gz']
    image_names = ['91493466_PROC_L_MLO_20190527195023.nii.gz']
    for _, image_name in enumerate(image_names):
        i=2
        input_file = f"../media/final_input/{image_name}"
        mask_file = f"../media/final_output/{image_name}"
        save_central_slice(input_file, mask_file, f"birads{i+1}_pred_baseline.png")
