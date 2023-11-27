.PHONY: help
DOCKER_IMG=jeremyadamsfisher1123/finetune-sdv:0.0.1
CONDA=micromamba

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

download:  ## download vox celeb
	@cog run python -W ignore load_videos_improved.py \
		--metadata vox-metadata.csv \
		--out_folder ./data \
		--workers 6 \
		--youtube 'yt-dlp'

lint:  ## clean up the source code
	@isort .
	@black .

build:  ## build the docker image
	@cog build -t $(DOCKER_IMG)

push:  ## push the docker image
	@cog push $(DOCKER_IMG)

run:  ## run something in docker
	@cog run \
		-e HF_DATASETS_CACHE="./hf_cache" \
		-e TRANSFORMERS_CACHE="./hf_cache" \
		-e HF_HOME="./hf_cache" \
		-e PYTHONPATH=. \
		-p 8888 \
		$(OPT)

lab:
	@$(MAKE) run OPT="python -m jupyterlab --allow-root --ip=0.0.0.0"

poke:  ## run interactive docker shell
	@$(MAKE) run OPT=bash

train:  ## run the training program
	@$(MAKE) run OPT="python -O sdvfinetune/train.py $(OPT)"

pip_freeze:
	@cp requirements.txt requirements.lock
	@cog run pip freeze > requirements.lock.tmp
	@rm requirements.lock
	@mv requirements.lock.tmp requirements.lock