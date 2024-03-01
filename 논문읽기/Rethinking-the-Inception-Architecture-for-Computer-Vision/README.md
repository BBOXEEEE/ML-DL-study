# Rethinking the Inception Architecture for Computer Vision (Inception-v2, v3)

### 0. Abstract

model size 를 늘리고, computational cost 를 늘리는 것이 성능 향상에는 좋지만,
computational efficiency 를 높이고, parameter 수를 줄이는 것은 여전히 mobile vision, big-data 처리를 가능하게 하는 요인이다.

저자는 **factorized convolution** 과 **aggressive regularization** 을 활용해 최대한 효율적으로 network 를 scale-up 하는 것을 목표로 한다.

### 1. Introduction

**AlexNet(2012)** 이 성공한 이후, 개선된 network 인  **VGGNet, GoogLeNet(2014)**  등이 많이 등장했다. 
흥미로운 점은 classifiaciton performance 가 개선되면, **computer vision** 의 다양한 분야도 함께 개선된다는 점이다. 

**VGGNet** 은 단순성이라는 강력한 기능을 가지고 있지만, 비용이 많이 든다.
반면, **GoogLeNet** 은 메모리와 computational budget 에 엄격한 제약 조건 속에서도 잘 동작하도록 설계되었다.
- GoogLeNet 은 AlexNet 에 비해 12배 적은 parameter 를 사용한다.
- VGGNet 은 AlexNet 보다 3배 더 많은 parameter 를 사용한다.

**Inception(GoogLeNet)** 아키텍처의 복잡성으로 인해 네트워크를 변경하기 어려워졌다.
성능을 키울 때, 단순히 scale-up 하는 것은 기존의 효율성을 잃는다. 그 예시로 단순히 filter size 를 2배로 늘리면 computational cost 와 parameter 는 4배로 증가한다. 또한, Inception 은 구조가 복잡해서 다양한 기법을 사용하면 더 복잡해진다.

따라서, 이 논문에서는 효율적인 방법으로 scaling-up 하는 몇가지 원칙과 최적화 아이디어를 소개한다.

### 2. General Design Principles

large-scale experimentation 을 기반으로 한 몇가지 design principels 를 소개한다.
이러한 원칙에서 크게 벗어나면 network 의 품질이 저하되는 경향이 있었고, 이것을 수정하면 일반적으로 아키텍처가 개선되었다.

1. **Avoid representational bottlenecks, especially early in the network.**
	-  Feed-forward network 는 정보 흐름의 방향을 제시한다.
	- extreme compression 을 통한 bottleneck 은 반드시 피해야한다.
	- representation size 는 입력에서 출력까지 **gently decrease** 되어야 한다.
	- 즉, feature map 을 급격하게 줄이지 말라는 의미로 해석할 수 있다. (feature map 을 급격하게 줄이면 정보 손실이 반드시 발생한다.)
2. **Higher dimensional representations are easier to process locally within a network**
	- 높은 차원의 표현이 network 내 지역적으로 처리하기 더 쉽다.
	- **activation** 횟수를 늘리면 feature 분리를 더 잘한다.
	- 그리고 이것이 학습 속도를 빠르게 한다.
3. **Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.**
	- spatial aggregation (conv 를 의미하는 것 같음) 을 할 때 차원 축소를 해도 representation power 손실 없이 lower dimensional embedding 을 할 수 있다.
	- conv 전 차원 축소를 해도 인접 unit 간 강한 상관관계로 인해 정보 손실을 줄여준다는 가설에 기반한 원칙이다.
	- conv 전 차원 축소는 학습 속도를 빠르게 한다.
4. **Balance the width and depth of the network.**
	- network 의 width & depth 를 늘리면 성능이 향상된다.
	- 최적의 improvement 를 위해서 parallel 하게 증가해야한다.
	- 따라서, computational budget 을 width 와 depth 에 균형있게 분배해야한다.

이러한 원칙은 모호한 상황에서 신중하게 적용했다.

### 3. Factorizing Convolutions with Larget Filter Size

