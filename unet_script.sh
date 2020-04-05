#!/bin/bash
python3 train_unet.py

python3 train_unet_m.py

cd playground/

python3 initialized.py
