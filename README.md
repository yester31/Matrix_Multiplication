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
            <td><strong>after Transpose</strong></td>
            <td><strong>OpenMP</strong></td>
            <td><strong>OpenMP & Transpose</strong></td>
            <td><strong>SSE</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>434 ms</td>
			<td>2020 ms</td>
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
            <td><strong>Naive</strong></td>
            <td><strong>with Shared Memory</strong></td>
            <td><strong>Cublas</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>13 ms</td>
			<td>16 ms</td>
			<td>4 ms</td>
		</tr>
	</tbody>
</table>


## Description
- ...