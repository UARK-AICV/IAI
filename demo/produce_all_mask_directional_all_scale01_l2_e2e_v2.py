import open_clip
import cv2 
import SimpleITK as sitk
import numpy as np
import pickle
import pandas as pd
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import numpy as np
import os
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from iai.model import TIAIBiomedCLIPL2_e2e_v2

def read_dicoms(image_path):
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)
    image = image/np.max(image)
    image = np.moveaxis(image, 0, -1)
    image = image.repeat(3, axis=-1)
    image = Image.fromarray((image * 255).astype(np.uint8)).resize((224, 224))
    return image

def plot_jet(image, mask):
    mask2 = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET) 
    mask2 = mask2.astype(float) + image.astype(float)
    mask2 = mask2 / mask2.max() * 255.0
    mask2 = mask2.astype(np.uint8)
    mask2[:, :, [2, 0]] = mask2[:, :, [0, 2]]
    return Image.fromarray(mask2)

with open("config file here", 'rb') as f:
    cfg = pickle.load(f)

model = build_model(cfg)  # returns a torch.nn.Module
DetectionCheckpointer(model).load("data_thang_personal/IAI/output/tIAI_with_static_heatmap_full_transcript_minianatomic_biomedclip_l2_mask_dice_heatmap_align_corners_scale01_e2e_v2_halfheart/model_final.pth")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()
_, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# NOTE: After we are done with the pipeline classifier, we will run the version we trained last night.
full_data_path = 'data_thang_personal/SAMed_materials/output_fixed_all.csv'
full_data = pd.read_csv(full_data_path)
root_ = '/home/tp030/IAI/data_thang_personal/mimic-cxr-2.0.0.physionet.org'
default_prompt = ["","possible left basilar opacity.", "possible right basilar opacity.", "the heart is enlarged."] # 0 = empty, 1,2,3
prefix = "data_thang_personal/IAI/output/IAI_heatmaps_left_right_directional_all_l2_align_corners_scale01_e2e_v2_halfheart"
os.makedirs(prefix, exist_ok=True)

for index, row in tqdm(full_data.iterrows(), total=len(full_data)):
    reflacx_id = row['reflacx_id']
    image_path = row['path']
    image_raw = read_dicoms(os.path.join(root_, image_path))
    image = preprocess_val(image_raw)
    normal_image = read_dicoms(os.path.join(root_, 'files/p17/p17848101/s54509320/830ad997-4d4c1e2a-ea00d4c3-12a4c704-56c5140b.dcm'))
    normal_image = preprocess_val(normal_image)
    batched_inputs = [{"meta":{"dataset_name": "TIAI_gaze_test_e2e_v2"}, "image": image, "normal_image":normal_image,"label": torch.tensor([1]), "transcript": 'left lung and right lung and the heart'}]
    batched_inputs.append({"meta":{"dataset_name": "TIAI_gaze_test_e2e_v2"}, "image": image,"normal_image":normal_image,"label": torch.tensor([1]),  "transcript": 'left lung'})
    batched_inputs.append({"meta":{"dataset_name": "TIAI_gaze_test_e2e_v2"}, "image": image,"normal_image":normal_image,"label": torch.tensor([1]),  "transcript": 'right lung'})
    batched_inputs.append({"meta":{"dataset_name": "TIAI_gaze_test_e2e_v2"}, "image": image,"normal_image":normal_image,"label": torch.tensor([1]),  "transcript": 'the heart'})

    outputs = [] 
    for batched_input in batched_inputs:
        output = model([batched_input])[0]
        outputs.append(output)

    # ok now save all the masks, the folder is /home/tp030/IAI/data_thang_personal/home/tp030/IAI_heatmaps_cardiac_left_right. The file name is join of `root + reflacx_id + _ + full/left/right/cardiac + .png`
    # NOTE: We may need to interpolate into the original image size in the future, but not for now. As we will stick with 224x224. `F.interpolate(torch.from_numpy(a[None]).unsqueeze(0).float(), size=(512,512), mode='bilinear', align_corners=True).squeeze(0).numpy().shape`
    
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_mask_full.png"), (outputs[0]["sem_seg"].argmax(dim=0).detach().cpu().numpy()*255).astype(np.uint8))
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_mask_left.png"), (outputs[1]["sem_seg"].argmax(dim=0).detach().cpu().numpy()*255).astype(np.uint8))
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_mask_right.png"), (outputs[2]["sem_seg"].argmax(dim=0).detach().cpu().numpy()*255).astype(np.uint8))
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_mask_cardiac.png"), (outputs[3]["sem_seg"].argmax(dim=0).detach().cpu().numpy()*255).astype(np.uint8))

    cv2.imwrite(os.path.join(prefix, reflacx_id + "_heatmap_full.png"), (outputs[0]["heatmap"][-1].detach().cpu().numpy()*255).astype(np.uint8))
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_heatmap_left.png"), (outputs[1]["heatmap"][-1].detach().cpu().numpy()*255).astype(np.uint8))
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_heatmap_right.png"), (outputs[2]["heatmap"][-1].detach().cpu().numpy()*255).astype(np.uint8))
    cv2.imwrite(os.path.join(prefix, reflacx_id + "_heatmap_cardiac.png"), (outputs[3]["heatmap"][-1].detach().cpu().numpy()*255).astype(np.uint8))


    # break
    del batched_inputs
    del outputs
    del image_raw
    del image

