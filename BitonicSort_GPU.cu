#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include<stdio.h>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>
#include<iterator>

using namespace std;

#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif 

// structure for trajectory
struct trajectory {
	// a struct containing all the points consistent in a trajectory
	vector<int> trajectoryPoints;
	float* distance = new float[0];
};

// global function for the GPU to calculate the total distance in the entire trajectory
__global__ void euclidean(int* points, int stride, float* distArray, float* distance, int length) {
	int i = threadIdx.x;
	int index = i * stride;
	int squaredSum = 0;

	// allow multi-dimensional
	for (int j = 0; j < stride; j++) {
		squaredSum += (points[index + j] - points[stride + index + j]) * (points[index + j] - points[stride + index + j]);
	}
	//distArray[i] = hypotf((points[index] - points[stride + index]), (points[index + 1] - points[stride + index + 1]));
	distArray[i] = sqrtf(squaredSum);
	__syncthreads(); // sync all the threads before performing a reduction
	
	// reduce
	float sum = 0;
	if (threadIdx.x == 0) {
		for (int k = 0;k < length;k++) {
			sum += distArray[k];
		}
		*distance = sum;
	}
}

__global__ static void bitonicsort(float * values, int lineNos)
{
	extern __shared__ float shared[];
	const unsigned int tid = threadIdx.x;
	shared[tid] = values[tid];
	__syncthreads();
	for (unsigned int k = 2;k <= lineNos ;k *= 2) {
		for (unsigned int j = k / 2; j > 0; j /= 2)
		{
			unsigned int ixj = tid ^ j;

			if (ixj > tid) {
				if ((tid & k) == 0)
				{
					if (shared[tid] > shared[ixj])
					{
						float temp = shared[tid];
						shared[tid] = shared[ixj];
						shared[ixj] = temp;
					}
				}

				else
				{
					if (shared[tid] < shared[ixj])
					{
						float temp = shared[tid];
						shared[tid] = shared[ixj];
						shared[ixj] = temp;
					}
				}
			}
			__syncthreads();
		}
	}
	values[tid] = shared[tid];
}

int main(int argc, char** argv)
{
	// variables used everywhere
	int num_of_stops = 0;
	int num_of_rows = 0;

	// accept commandline arguments or exit if none provided (filename, from the same folder)
	if (argv[1] == NULL || argv[1] == "") {
		printf("Error Reading File\n");
		exit(0);
	}

	// read file using ifstream
	std::ifstream myFile(argv[1]);

	vector<float> values;
	vector<trajectory> trajectories;
	float * d_values;

	/*while (!std::feof(myFile)) {
		std::getline(myFile);
		fscanf(myFile, "%d\n", &values[i]);
		for (int j = 0; j < ) {

		}
	}*/

	string line = "";

	// open file and start parsing line by line
	while (getline(myFile, line))
	{
		trajectory trajTemp;

		stringstream lineTokens(line);
		string temp; // breaking line into string
		vector<int> temp1;
		// parse line and extract numbers from the line
		while (getline(lineTokens, temp, ' ')) {
			temp1.push_back(atoi(temp.c_str()));
		}

		if (2 == temp1.size()) {
			// from the first line, acquire the number of stops
			num_of_rows = temp1[0];
			num_of_stops = temp1[1];
		}
		else {
			// this applies for the lines after the first line
			// calculating stride
			int stride = temp1.size() / num_of_stops;

			// fill in the origin point
			for (int i = 0; i < stride; i++) {
				temp1.insert(temp1.begin(),0);
			}

			// calculating parallel euclidean distance using GPU for the entire trajectory
			// this performs an outer loop parallelization by loop distribution
			// i.e. I broke the Euclidean distance calculation and sorting in 2 different GPU operations
			// this ensures that calculation heavy operations are efficiently performed by the GPU.
			int* d_points;
			float* d_distArray;
			float* d_res;
			cudaMalloc(&d_points, temp1.size() * sizeof(int));
			cudaMalloc(&d_distArray, num_of_stops * sizeof(float));
			cudaMalloc(&d_res, sizeof(float));

			cudaMemcpy(d_points, temp1.data(), temp1.size() * sizeof(int), cudaMemcpyHostToDevice);
			
			euclidean << <1, num_of_stops >> > (d_points, stride, d_distArray, d_res, num_of_stops);

			cudaMemcpy(trajTemp.distance, d_res, sizeof(float), cudaMemcpyDeviceToHost);

			cudaFree(d_points);
			cudaFree(d_distArray);
			cudaFree(d_res);

			values.push_back(*trajTemp.distance);
			
			// add all points to the temporary trajectory vector
			trajTemp.trajectoryPoints = temp1;

			// push current trajectory to our trajectories vector
			trajectories.push_back(trajTemp);
		}
	}

	// close the input file
	myFile.close();

	cudaMalloc(&d_values, num_of_rows * sizeof(float));
	cudaMemcpy(d_values, values.data(), sizeof(float) * num_of_rows, cudaMemcpyHostToDevice);

	bitonicsort << <1, num_of_rows, num_of_rows * sizeof(float)>> > (d_values, num_of_rows);

	float* sortedValues = new float[num_of_rows];
	cudaMemcpy(sortedValues, d_values, sizeof(float) * num_of_rows, cudaMemcpyDeviceToHost);
	
	// uncomment this snippet to debug the code and see output of the sorted array coming from the GPU
	/*printf("\n---------------------------------\n");
	printf("Here is the sorted array from GPU: \n");
	for (int i = 0; i < num_of_rows; i++)
	{
		printf("%f, ", sortedValues[i]);
	}
	printf("\n---------------------------------\n");*/

	// write the output to a file
	std::ofstream output_file("./output.txt");
	for (int i = 0; i < num_of_rows; i++)
	{
		// the trajectories are not sorte in the data structure but we now have the sorted array
		vector<int> trajectoryToPrint;
		for (int j = 0; j < trajectories.size(); j++) {
			if (*trajectories[j].distance == sortedValues[i]) {
				trajectoryToPrint = trajectories[j].trajectoryPoints;
			}
		}
		// write vector to file
		ostream_iterator<int> output_iterator(output_file, " ");
		copy(trajectoryToPrint.begin()+2, trajectoryToPrint.end(), output_iterator);
		output_file << "\n";
	}

	// free memory
	cudaFree(d_values);
	free(sortedValues);
}
