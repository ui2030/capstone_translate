# Cocktail
현 프로젝트는 인공지능 모델을 기반으로 실시간 번역을 위해 만들어졌다.
실시간으로 인식하기 위해서 OCR이 필요한데, 여기에 필요한 OCR로는 tesseract을 사용하였다.
원래는 tesseract, Paddle, Easy OCR들을 사용하려했지만, 활용을 제대로 하지못해 1개 쓰는것보다 못한 성능으로 결국 하나만 사용하였다.
번역 모델로는 Hugging Face에서 사용할 수 있는 facebook/m2m100_418M을 사용하였다.
이름이 나와있듯이 facebook을 기반으로 만들어진 모델이며, 다중 언어가 지원되는 모델이라 약 80여개의 언어들을 번역 시켜볼 수 있다.
하지만 본 프로젝트에서는 오직 영어 -> 한국어만을 다룬다.
파인튜닝(Fine-tuning)을 통해 영어에서 한국어로 통역하는 기능을 향상시켰다.
그 외에 더 발전시킬 여지가 있다면 인식되는 언어 -> 영어(중간 언어) -> 원하는 언어로 거쳐가는 형태를 생각하고있다.

아직 최적화를 거치지못해 상당히 느리다.
