#!/bin/bash
# +
# python3 train_unet.py

# +
# python3 train_unet_m.py
# -

python3 train_unet_m_sumloss.py

cd scripts/

python3 initialized.py
