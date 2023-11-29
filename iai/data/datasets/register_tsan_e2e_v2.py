import os
import random
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
import open_clip
from iai.utils.misc import read_gt_heatmap
from torchvision import transforms

def map_type_namecol(_type):
    name_col = ''
    if _type == 'c':
        name_col = 'mask (cardiomegaly)'
    if _type == 'l':
        name_col = 'mask (left lung)'
    if _type == 'r':
        name_col = 'mask (right lung)'
    return name_col

def map_type_id(_type):
    if _type == 'c': return '3'
    if _type == 'l': return '1'
    if _type == 'r': return '2'

def map_type_label(_type):
    if _type == 'c': return 'severity'
    if _type == 'l': return 'has finding'
    if _type == 'r': return 'has finding'

def map_type_default_sentence(_type):
    if _type == 'c': return 'the heart'
    if _type == 'l': return 'left lung'
    if _type == 'r': return 'right lung'

def Tiai_gaze_function(df):
    # change this function into for looping all image type, if there is left then add left item, if there is right then add right item, if there is cardiac then add cardiac item, and finally add the static heatmap item
    mimic_cxr_path = '/home/tp030/iai/data_thang_personal/mimic-cxr-2.0.0.physionet.org'
    mapping = {0:0, 1:1, 2:1, 3:1, 4:1}
    d_dicts = []
    break_point = 3
    for i, row in df.iterrows():
        _type = row['_type']
        name_col = map_type_namecol(_type)
        name_col_heatmap = name_col.replace('mask', 'heatmap')
        name_col_sentence = name_col.replace('mask', 'sentence')
        name_col_label = map_type_label(_type)
        default_sentence =  map_type_default_sentence(_type)
        # res = {}
        # res['normal_image_dcm'] = os.path.join(mimic_cxr_path, 'files/p17/p17848101/s54509320/830ad997-4d4c1e2a-ea00d4c3-12a4c704-56c5140b.dcm')
        # reflacx_id = row['reflacx_id']
        # image_dcm = row['path']
        # res['image_dcm'] = os.path.join(mimic_cxr_path, image_dcm)
        # res['gt_heatmap_path'] = row[name_col_heatmap]
        # res['gt_mask_path'] = row[name_col]
        # res['reflacx_id'] = reflacx_id
        # res['heatmap_type'] = _type.upper()+"r"
        # res['transcript'] = row[name_col_sentence]
        # res['label'] = mapping[int(row[name_col_label])]
        # d_dicts.append(res)

        res = {}
        res['normal_image_dcm'] = os.path.join(mimic_cxr_path, 'files/p17/p17848101/s54509320/830ad997-4d4c1e2a-ea00d4c3-12a4c704-56c5140b.dcm')
        reflacx_id = row['reflacx_id']
        image_dcm = row['path']
        res['image_dcm'] = os.path.join(mimic_cxr_path, image_dcm)
        res['gt_heatmap_path'] = row[name_col_heatmap]
        res['gt_mask_path'] = row[name_col]
        res['reflacx_id'] = reflacx_id
        res['heatmap_type'] = _type.upper()
        res['transcript'] = default_sentence
        res['label'] = mapping[int(row[name_col_label])]
        d_dicts.append(res)

    return d_dicts


def register_all_tiai(root):
    train_df = pd.read_csv("/home/tp030/iai/data_thang_personal/physionet.org/files/reflacx_unet_real_label_with_transcript_reflacx_full_train_42_minianatomic_severity_mix_v2.csv")
    dev_df = pd.read_csv("/home/tp030/iai/data_thang_personal/physionet.org/files/reflacx_unet_real_label_with_transcript_reflacx_full_dev_42_minianatomic_severity_mix_v2.csv")
    test_df = pd.read_csv("/home/tp030/iai/data_thang_personal/physionet.org/files/reflacx_unet_real_label_with_transcript_reflacx_full_test_42_minianatomic_severity_mix_v2.csv")
    

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    input_size = preprocess_train.transforms[0].size[0]
    preprocess_heatmap = transforms.Compose([transforms.Resize([input_size, input_size]), transforms.ToTensor()])
    for name, df, preprocess in [
        ("train", train_df, preprocess_train),
        ("test", dev_df, preprocess_val),
    ]:
        data_name = f"Tiai_gaze_{name}_e2e_v2"
        DatasetCatalog.register(
            data_name,
            lambda x=df: Tiai_gaze_function(x),
        )
        MetadataCatalog.get(data_name).set(
            label_root='/home/tp030/iai/data_thang_personal/physionet.org/files/reflacx_unet_real_label_with_transcript_reflacx_full_42_minianatomic_severity_mix_v2.csv',
            evaluator_type="Tiai_gaze_e2e_v2",
            ignore_label=255,
            thing_classes = ['NotInterest', 'Interest'],
            preprocess=preprocess,
            preprocess_heatmap=preprocess_heatmap,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_tiai(_root)
