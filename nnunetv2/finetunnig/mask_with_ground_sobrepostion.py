import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def save_central_slice(input_file, pred_mask_file, gt_mask_file, output_file="overlay.png"):
    # carregar imagem e máscaras
    image = nib.load(input_file).get_fdata()
    pred_mask = nib.load(pred_mask_file).get_fdata()
    gt_mask = nib.load(gt_mask_file).get_fdata()

    # normalizar imagem para 0-1 (melhor contraste)
    image = image.astype(np.float32)
    image = image / np.max(image)

    # escolher slice central
    z = image.shape[2] // 2
    slice_image = image[:, :, z]
    slice_pred_mask = pred_mask[:, :, z]
    slice_gt_mask = gt_mask[:, :, z]
    
    # aplicar rotação e flip
    slice_image = np.flip(np.rot90(slice_image), axis=1)
    slice_pred_mask = np.flip(np.rot90(slice_pred_mask), axis=1)
    slice_gt_mask = np.flip(np.rot90(slice_gt_mask), axis=1)
    
    # plotar e salvar
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_image, cmap="gray")
    
    # sobreposição do ground truth (ex: verde)
    plt.imshow(slice_gt_mask, cmap="Greens", alpha=0.4)
    
    # sobreposição da predição (ex: vermelho)
    plt.imshow(slice_pred_mask, cmap="Reds", alpha=0.4)
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    print(f"Salvo em {output_file}")

if __name__ == "__main__":
    # Criar pasta para salvar as imagens
    output_dir = "sobreposicao_imgs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Exemplo de uso
    image_names = ['4537110_PROC_L_MLO_20130105193311.nii.gz',
                  '76201127_PROC_L_MLO_20150105143403.nii.gz',
                  '91493466_PROC_L_MLO_20190527195023.nii.gz',
                  '92463763_PROC_L_MLO_20200302182630.nii.gz']

    for i, image_name in enumerate(image_names):
        input_file = f"../media/final_input/{image_name}"
        pred_mask_file = f"../media/final_output/{image_name}"  # máscara prevista
        gt_mask_file = f"../media/ground_truth/{image_name}"    # máscara ground truth
        
        output_file = os.path.join(output_dir, f"birads{i+1}_sobreposicao.png")
        save_central_slice(input_file, pred_mask_file, gt_mask_file, output_file)