# Deep Residual Learning for Image Recognition (ResNet)

### 0. Abstract

딥러닝에서 Neural Network 의 깊이가 깊어질수록 성능은 좋지만 학습이 어렵다. 그래서 이 논문에서 잔차를 이용한 `Residual Learning Framework` 를 통해서 깊은 신경망에서도 학습이 쉽게 이루어질 수 있다는 방법론을 제시한다.

> We explicitly reformulate the layer as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.

- 함수를 새로 만드는 것 대신 `Residual Function` 을 학습에 사용하는 것으로 layer 를 재구성한다.

`Empirical evidence showing` 방법으로 **Residual Network** 가 optimize 하기 쉽고, layer 를 깊게 쌓았을 때에도 accuracy 를 증가시키는 방법에 제공한다.
- empirical : 경험적인 방식으로 데이터를 이용한 실험을 통해 증명하는 방식

### 1. Introduction

많은 논문들을 통해 `Deep Network` 일수록 성능이 좋다는 것은 알려져 있는 사실이다.


> Is learning better networks as easy as stacking more layers?

- 깊어지는 layer 와 함께 떠오르는 의문은 **layer 를 더 많이 쌓는 것만큼 성능이 좋아질까?** 이다.

>Normalized initialization and intermediate normalization layers, which enable networks with tens of layers to start converging for stochastic gradient descent (SGD) with backpropagation.

- 이와 관련해 **vanishing/exploding gradients** 문제가 발생하는데 그래도 이러한 문제는 다양한 방법으로 해결되어 왔다.

이 논문에서 깊게 다루는 문제는 **Degradation Problem** 이다.
- network 가 깊어질수록 accuracy 떨어지는 문제이다.
![[ResNet FIg.1.png]]

흥미로운 점은 **overfitting** 으로 발생하는 문제가 아니라는 것이다. 그 이유는 overfitting 에 의한 문제라면, training accuracy 는 높고, test accuracy 는 낮아야하는데 Fig. 1 에 제시된 그래프에 의하면 두개 다 낮기 때문이다. 따라서, 이 논문에서는 깊은 layer 가 쌓일 수록 optimize 가 복잡해져 발생하는 부작용으로 인식하고 해결하려 한다.

**Identity mapping layer** 를 추가했지만, 실험적으로 좋은 solution 은 아니라고 결과를 얻었고, **Deep Residual Learning Framework** 라는 개념을 도입한다.

> Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping.

- 쌓여진 layer 가 바로 다음 layer 에 적합하는 것이 아닌, 잔차의 mapping 에 적합하도록 만들었다.

이 논문에서 nonlinear layers fit 인 `F(x) = H(x) - x` 를 제시하고, 이것을 전개하면 `H(x) = x + F(x)` 가 된다. 여기서 **residual mapping** 이 기존의 mapping 보다 optimize 하기 쉽다는 것을 가정한다.
- H(x) = F(x) 라면, 이것은 Identity Matrix 에 수렴해야한다.
- H(x) = F(x) + x 라면, 이것은 zero 에 수렴해야한다.
	- 즉, **residual** 을 **zero** 로 만드는 것이 더 쉽다는 것이다.

![[ResNet Fig.2.png]]

`F(x) + x` 는 **Shortcut connection** 과 동일한데, 이는 하나 또는 이상의 layer 를 skip 하게 만들어준다.
위 그림에서 x 는 input, F(x) 과정을 거쳐 identity 인 x가 더해져 output 으로 F(x) + x 가 나오는 형태이다.

이 논문의 2가지 목표는 다음과 같다. **(쉬운 optimize & accuracy 향상)**

> Our extremely deep residual nets are easy to optimize, but the counterpart "plain" nets (that simply stack layers) exhibits higher training error when the depth increases.

- plain net 보다 더 쉽게 optimize

> Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.

- 더 쉽게 accuracy 향상

### 2. Related Work

**Residual Representation** 과 **Shortcut Connection** 개념에 대한 논문들에 대한 간략한 설명과 이것들에 비해 **Residual Network** 가 가지는 장점에 대해 설명한다.

### 3. Deep Residual Learning

#### 3.1 Residual Learning

Introduction 에서 설명한 `H(x) = F(x) + x` 에 대해 설명한다.
H(x) 를 기본 mapping 이라고 간주하고, x 가 input 일 때, 여러 비선형 layer 가 복잡한 함수를 근사할 수 있다고 가정하면 잔차 함수도 점근적으로 근사할 수 있다는 가설을 세우는 것은 동일하다.

다만, 잔차함수 `F(x) = H(x) - x` 에 근접하도록 허용하고, 따라서 원래 함수는 `F(x) + x` 가 된다.
형태에 따라 학습의 용이성은 다를 수 있다고 말한다.

즉, 정리하자면 다음과 같다.
![[ResNet img01.png]]

- **without skip-connection**, **xW1W2** 는 Identity Matrix 에 적합해야한다.

![[ResNet img02.png]]

- **with skip-connection**, **xW1W2** 는 **zero** 에 적합해야한다.

결국, 두 방법 중 **residual** 을 이용하는 것이 학습의 용이성이 더욱 좋다는 것이다.

#### 3.2 Idendity Mapping by Shortcuts

**Shortcut connection은 파라미터나 연산 복잡성을 추가하지 않는다.** 이때, F + x 연산을 위해 x와 F의 차원이 같아야 하는데, 이들이 서로 다를 경우 linear projection 인 Ws 를 곱하여 차원을 같게 만들 수 있다. 여기서 Ws 는 차원을 매칭 시켜줄 때에만 사용한다.

#### 3.3 Network Architectures

**Plain Network**
baseline 모델로 사용한 plain net 은 VGGNet 에서 영감을 얻었다. conv size 가 3x3 이고, 다음 2가지 규칙에 의해 설계했다.
- Output feature map size 가 동일한 layer 들은 **같은 수의 conv filter 를 사용** 한다.
- 만약, feature map 이 절반으로 줄어들면 **time complexity** 를 동일하게 유지하기 위해 **filter 의 수를 2배로 늘려준다.**

downsampling 을 수행하면, pooling 을 사용하는 것이 아닌 **stride 가 2인 conv filter 를 사용한다.** 모델의 끝단은 **GAP를 사용** 하고, 사이즈가 **1000인 FC layer 와 Softmax 를 사용** 한다. 결과적으로 총 34개의 layer 가 사용되며, VGGNet 보다 적은 필터와 복잡성을 가진다.

**Residual Network**

plain 모델에 기반하여 **Shortcut Connection** 을 추가해 구성한다. 이때, input 과 output 의 dimension 이 동일하면 그대로 적용하면 되지만, 다를 경우 2가지 option 이 있다.
- **zero padding** 을 통해 dimension 을 키운다.
- **1 x 1 conv** 인 **projection shortcut** 을 적용한다.

2가지 option 모두 feature map 을 2 size 씩 건너뛰므로 **stride 는 2로 수행** 한다.

#### 3.4 Implementation

모델 구현은 다음과 같이 진행

1. 짧은 쪽이 [256, 480] 이 되도록 random 하게 resize
2. 224 x 224 사이즈로 random 하게 crop
3. Horizontal flip 을 부분적으로 적용하고 per-pixel mean 을 뺌
4. Standard color augmentation 적용
5. Z (affine sum) 에 BN 적용
6. Initialized Weight
7. Optimizer 로 SGD 사용, mini-batch size 는 256
8. Learning rate 는 0.1 에서 시작하고, 학습이 정체될 때마다 10씩 나눔
9. Weight decay 는 0.0001, Momentum 은 0.9
10. Dropout 은 적용 X

테스트는 10-cross validation 방식 적용, multiple scale 을 적용해 짧은 쪽이 {224, 256, 384, 480, 640} 중 하나가 되도록 resize 후 평균 score 산출

### 4. Experiments

#### 4.1 ImageNet Classification

**Plain Networks**

18-layer 와 34-layer 로 실험을 진행, 결과적으로 38-layer 의 training error 가 더 높게 나왔다. 이러한 문제가 발생하는 원인이 vanishing gradient 문제가 발생한 것은 아니라고 주장했으며 deep 한 plain 모델은 **exponentially low convergence rate** 문제를 가지고 있을 것이라 설명했다.

**Residual Networks**

Plain Networks 와 마찬가지로 18-layer, 34-layer 로 실험을 진행했고, **shortcut connection** 하는 2가지 option 중 **zero-padding** 방식을 추천했다.

ResNet 실험에서 중요한 3가지 사실을 발견했다. 
1. 34-layer 에서 **degradation problem** 을 피할 수 있어 성능이 증가했다는 점이다.
2. **residual learning** 을 적용하면 layer 를 더 깊게 쌓고 성능을 높일 수 있다는 점이다.
3. ResNet 이 조금 더 빨리 최고 성능에 도달한다는 점이다.


앞에서, skip-connection 하는 2가지 option 중 zero-padding 방식을 이용했는데, 다른 option 을 적용해본다.
1. zero-padding 은 parameter 가 증가하지 않는다.
2. projection shortcut 은 1 x 1 conv filter 를 추가하기 때문에 parameter 가 약간 증가한다.
3. 모든 skip-connection 이 진행되는 단계에 projection 을 적용해준다.

결과적으로 성능은 `1 < 2 < 3` 이지만, 성능차이가 미비했고, parameter 수 차원에서 3번째 방법은 피했다고 설명했다.

#### 4.2 CIFAR-10 and Analysis

CIFAR-10 데이터로 실험한 결과도 제시한다. 결과적으로 좋은 성능을 보여주고 있고, 1000개 이상의 layer 도 쌓아보았는데 이때 성능이 낮아지는 것은 overfitting 문제이다.