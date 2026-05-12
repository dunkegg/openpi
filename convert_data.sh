export SRC=data/lerobot/express_eval
export DST=data/lerobot/express_eval_v3

python convert_raw2v3.py --src "$SRC" --dst "$DST" --overwrite


export SRC=data/lerobot/express_eval_v3
export DST=data/lerobot/express_eval_v2pi

python convert_v3_to_pi05_v2.py --src "$SRC" --dst "$DST"
