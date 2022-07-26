# Information-Retrieval-dosilt

한국어 데이터에 대해서 Information Retrieval 모델에 대해서 개발한 기록입니다. 

2022 인공지능 온라인 경진대회에 이용했던 결과이며, 실제 테스트 데이터에서 적용한 결과가 아닌  
train 데이터에 80%는 학습에 20%는 테스트에 이용한 결과이며, 데이터는 비공개 사항이라 공개하진 않았습니다. 

대략적인 알고리즘만 봐주시면 좋을 것 같으며, **코드가 틀린 부분이 존재할 수 있습니다.**

bm25 : space를 기준으로 DTM를 생성한 결과, Konlpy의 Okt를 활용하여 DTM을 생성한 결과 

bm25+rerank : train 데이터의 bm25결과를 이용하여 특정 query와 연관성이 높은 상위 10개 document를 이용하여 학습

ColBERT : ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT  


| |모델|MRR|설명|
|--|------|---|---|
|**1**|bm25+space|0.7815|space(공백)을 기반으로 DTM을 생성하였습니다.|
|**2**|bm25+Okt|0.8734|Okt의 morphs를 이용하여 DTM을 생성하였습니다.|
|**3**|**1**+rerank|0.9199|KoElectra-small 모델을 기반으로 학습하였습니다.|
|**4**|**2**+rerank|0.9873|KoElectra-small 모델을 기반으로 학습하였습니다.|
|**6**|ColBERT|0.9524|KoElectra-base 모델을 기반으로 학습하였습니다.|
|**7**|**1**+ColBERT|0.9198|**6** 과 동일한 모델을 이용하였습니다.|
|**8**|**2**+ColBERT|0.9872|**6** 과 동일한 모델을 이용하였습니다.|  




# 추가적인 개선사항
> *1. 학습을 좀 더 길게*
> > 현재 간단한 구현 확인만 하기 위해 epoch은 4로 학습, loss graph로 보건데 수렴이 덜 된것으로 보임  
>
> *2. 다양한 데이터로 학습*
> > 학습 데이터를 bm25의 결과의 상위 n개로 제한 두지 말고 넓게 학습  
>
> *3. 하이퍼 파라미터 탐색*
> > 최적의 모델 성능을 끌어내기 위한 파라미터 탐색 과정 (ex, LR, Initializer, Epoch, Batch, Data preprocessing etc..)
