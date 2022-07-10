# Matrix_Multiplication

## Enviroments
* Windows 10 laptop
* CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)
* NVIDIA GeForce RTX 3060 Laptop GPU (gpu)
* opencv Version : 3.4.5

## Performace Evaluation (Release mode)
* A[1024, 1024] * B[1024, 1024] = C[1024, 1024]
* 100 iteration

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>cpu</td>
			<td><strong>OpenCV</strong></td>
            <td><strong>Naive</strong></td>
            <td>after Transpose</td>
            <td>OpenMP</td>
            <td>OpenMP & Transpose</td>
            <td>SSE</td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td><strong>434 ms</strong></td>
			<td><strong>2020 ms</strong></td>
			<td>930 ms</td>
			<td>419 ms</td>
			<td>139 ms</td>
			<td>452 ms</td>
		</tr>
	</tbody>
</table>

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>gpu</td>
            <td>CUDA</td>
            <td>CUDA with Shared Memory</td>
            <td><strong>Cublas</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>13 ms</td>
			<td>16 ms</td>
			<td><strong>4 ms</strong></td>
		</tr>
	</tbody>
</table>


## Description
- 여러 가지 방법을 사용하여 행렬 곱을 구현함.
- after Transpose : Naive 행렬곱 계산 과정을 보면 두 행렬에서 값을 가져올 때(메모리에 접근할 때) 하나의 행렬에서 연속적으로 접근하지 못하는 것을 볼 수 있다.
그렇기 때문에 매번 메모리 접근하기 위해 비용이 사용된다. 이를 개선하기 위해 사전에 전치 행렬(Transpose)로 변환하여 메모리 접근을 연속하게 만들어
메모리 접근에 사용되는 비용을 최소화하게 만듦.
- OpenMP : OpenMP를 이용하여 멀티 스레드로 계산, 여기서도 전치행렬을 사용하여 OpenMP만을 사용한 경우보다 더 나은 성능 개선을 얻음.
- SSE : Inter SSE를 사용하여 Float 4개씩 한 번에 처리하는 행렬곱 구현.
- CUDA : CUDA kernel 함수를 구현하여 gpu에서 병렬 처리 구현.
- CUDA with Shared Memory : Gpu에서 사용 가능한 메모리 공간 중 하나인 shared memory를 사용하여 메모리 접근성 개선함(coalesced access)
- Cublas : 넘사벽...
- 계산 결과에서 볼 수 있듯이 (병렬처리가 가능한 계산에서) GPU의 연산 속도는 CPU 보다 월등히 높다. 
하지만, GPU와 CPU는 서로 다른 메모리 공간을 갖는 Heterogeneous 한 구조를 갖기 때문에 데이터 전달 작업이 필요하다.
CPU가 사용하는 메모리 공간에서 GPU가 사용하는 메모리 공간으로 데이터를 전달해야 한다.
이 데이터 전달 비용이 생각 보다 크다. 그렇기 때문에 만약 작은 행렬 곱의 연산의 경우는 데이터 전달 시간 때문에 CPU에서 계산 속도가 월등히 나을 것이다.
연산의 크기에 맞게 적절한 함수를 선택할 수 있는 형태로 구현되어야 한다. 