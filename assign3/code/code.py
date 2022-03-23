!./asr.sh --stage 1 --stop_stage 1 --train_set "train" --valid_set "dev" --test_sets "dev test"  

!./asr.sh --stage 2 --stop_stage 2 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1"

!./asr.sh --stage 3 --stop_stage 3 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1" --audio_format "flac"

!./asr.sh --stage 4 --stop_stage 4 --nj 10 --max_wav_duration 15 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1" --lm_train_text "data/train/text"

!./asr.sh --stage 5 --stop_stage 5 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --token_type bpe --nbpe 5750 --bpemode "unigram" --bpe_train_text "data/train/text"

!./asr.sh --stage 6 --stop_stage 6 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --token_type bpe --nbpe 5750 --bpemode "unigram" --bpe_train_text "data/train/text" --lm_config "conf/train_lm_transformer.yaml"

!./asr.sh --stage 7 --stop_stage 7 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --token_type bpe --nbpe 5750 --bpemode "unigram" --bpe_train_text "data/train/text" --lm_config "conf/train_lm_transformer.yaml"

!./asr.sh --stage 8 --stop_stage 8 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --token_type bpe --nbpe 5750 --bpemode "unigram" --bpe_train_text "data/train/text" --lm_config "conf/train_lm_transformer.yaml"

!./asr.sh --stage 9 --stop_stage 9 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --token_type bpe --nbpe 5750 --bpemode "unigram" --bpe_train_text "data/train/text" --lm_config "conf/train_lm_transformer.yaml"

!./asr.sh --stage 9 --stop_stage 10 --nj 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --token_type bpe --nbpe 5750 --bpemode "unigram" --bpe_train_text "data/train/text" --lm_config "conf/train_lm_transformer.yaml"

!sed -i -e '1s/^/use_amp: true \n/' \
        -e '1s/^/cudnn_deterministic: false \n/' \
        -e '1s/^/cudnn_benchmark: false \n/' \
        -e 's/10000000/2500000/' conf/train_asr_conformer.yaml

!./asr.sh --stage 10 --stop_stage 10 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1" --nj 10 --token_type bpe --nbpe 5750 --asr_config "conf/train_asr_conformer.yaml" --inference_config 'conf/decode_asr.yaml'

!./asr.sh --stage 11 --stop_stage 11 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1" --nj 10 --token_type bpe --nbpe 5750 --ngpu 1 --asr_config "conf/train_asr_conformer.yaml"  --inference_config 'conf/decode_asr.yaml'

!./asr.sh --inference_nj 10 --stage 12 --stop_stage 12 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1" --token_type bpe --nbpe 5750 --asr_config "conf/train_asr_conformer.yaml" --use_lm true  --gpu_inference true   --inference_config 'conf/decode_asr.yaml' --lm_config "conf/train_lm_transformer.yaml"

!./asr.sh --inference_nj 10 --stage 13 --stop_stage 13 --train_set "train" --valid_set "dev" --test_sets "dev test" --speed_perturb_factors "0.9 1.0 1.1" --token_type bpe --nbpe 5750 --asr_config "conf/train_asr_conformer.yaml" --use_lm false --gpu_inference true   --inference_config 'conf/decode_asr.yaml' 

