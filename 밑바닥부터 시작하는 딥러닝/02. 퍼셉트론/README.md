# 02. 퍼셉트론

### 퍼셉트론이란?

**퍼셉트론 (perceptron)** 은 다수의 신호를 입력으로 받아 하나의 신호를 출력한다.

![](./imgs/perceptron.png)

- 입력으로 2개의 신호를 받은 퍼셉트론의 예시
- x1, x2 는 입력신호, w1, w2 는 가중치(weight) 이다.
- 입력 신호가 뉴런에 보내질 때 각각 고유한 가중치가 곱해지며, 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력한다.
- 이것을 뉴런이 활성화한다고 표현한다.

퍼셉트론은 복수의 입력 신호 각각에 고유한 가중치를 부여하는데, 가중치는 각 신호가 결과에 주는 영향력을 조절하는 요소로 작용한다. 즉, 가중치가 클수록 해당 신호가 그만큼 더 중요함을 의미한다.

### 단순한 논리 회로

AND, OR, NAND 게이트를 퍼셉트론으로 구현해본다.
퍼셉트론의 구조는 3가지 게이트에서 모두 똑같으며 가중치와 임계값의 차이만 있다.

```python
# AND
import numpy as np  

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(w*x) + b
	
	if tmp <= 0:
		return 0
	else:
		return 1

if __name__ == "__main__":
	for xs in [(0, 0), (1, 0), (0,1), (1, 1)]:
		y = AND(xs[0], xs[1])
		print(str(xs) + " -> " + str(y))

# NAND
def NAND(x1, x2):
	x = np.array([x1, x2 ])
	w = np.array([-0.5, -0.5])
	b = 0.7
	tmp = np.sum(w*x) + b
	
	if tmp <= 0:
		return 0
	else:
		return 1

if __name__ == "__main__":
	for xs in [(0, 0), (1, 0), (0,1), (1, 1)]:
		y = NAND(xs[0], xs[1])
		print(str(xs) + " -> " + str(y))

# OR
def OR(x1, x2):
	x = np.array([x1, x2 ])
	w = np.array([0.5, 0.5])
	b = -0.2
	tmp = np.sum(w*x) + b
	
	if tmp <= 0:
		return 0
	else:
		return 1

if __name__ == "__main__":
	for xs in [(0, 0), (1, 0), (0,1), (1, 1)]:
		y = OR(xs[0], xs[1])
		print(str(xs) + " -> " + str(y))
```

- `x` 는 입력신호이며, `w` 는 가중치, `b` 는 bias 이다.
- bias (편향) 은 뉴런이 얼마나 쉽게 활성화되는지를 결정한다.

### 퍼셉트론의 한계

퍼셉트론은 **XOR 게이트** 를 구현할 수 없다. 정확히 말하면 단층 퍼셉트론으로는 구현할 수 없다.
퍼셉트론은 직선으로 나뉜 두 영역을 만드는데, XOR 게이트를 두 구역으로 나누는 직선을 만들기는 불가능하다.

**비선형 영역** 즉, 곡선으로 나눈다면 두 구역으로 나눌 수 있다.
즉, **다층 퍼셉트론** 을 만들면 XOR 게이트를 구현할 수 있다는 말이다.

```python
from and_gate import AND
from nand_gate import NAND
from or_gate import OR

def XOR(x1, x2):
	s1 = NAND(x1, x2)
	s2 = OR(x1, x2)
	y = AND(s1, s2)
	return y

if __name__ == "__main__":
	for xs in [(0, 0), (1, 0), (0,1), (1, 1)]:
		y = XOR(xs[0], xs[1])
		print(str(xs) + " -> " + str(y))
```

- 앞에서 구현한 3가지 게이트를 조합하면 XOR 게이트를 만들 수 있다.

이와 같이 단층 퍼셉트론으로 표현하지 못한 것을 층을 하나 늘려 구현할 수 있다. 퍼셉트론은 층을 쌓아 더 다양한 것을 표현할 수 있다.

### 정리

- 퍼셉트론은 입출력을 갖춘 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다.
- 퍼셉트론에서는 **가중치** 와 **편항** 을 매개변수로 활용한다.
- 퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다.
- XOR 게이트는 단층 퍼셉트론으로는 표현할 수 없다.
- 2층 퍼셉트론을 이용하면 XOR 게이트를 표현할 수 있다.
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다.
- 다층 퍼셉트론은 이론상 컴퓨터를 표현할 수 있다.



