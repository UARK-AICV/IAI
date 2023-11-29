#!/bin/bash
mv env/lib/python3.8/site-packages/timm/models/vision_transformer.py env/lib/python3.8/site-packages/timm/models/vision_transformer_old.py
ln -s $(pwd)/notes/biomedclip_vs_openaiclip/patched_vision_transformer.py $(pwd)/env/lib/python3.8/site-packages/timm/models/vision_transformer.py