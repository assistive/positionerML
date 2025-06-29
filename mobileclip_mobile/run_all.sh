python scripts/download/download_models.py --models mobileclip_s0
python scripts/convert/convert_models.py --model mobileclip_s0 --platforms ios android
python scripts/deploy/deploy_mobile.py --zip
