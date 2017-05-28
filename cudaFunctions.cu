#include <stdio.h>
#include <cfloat>

const double voltsConstantMin = 0;
const double voltsConstantMax = 300;					//0-300V peak
const double omegaConstantMin = 2.0*M_PI*40;
const double omegaConstantMax = 2.0*M_PI*70;			//40-70Hz
const double phiConstant = 2.0*M_PI;					//0-2PI radians

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void runDEagent(const unsigned long int S, const unsigned long int G, const double F, const double R, const size_t N,
	double *bestScoreAndParameters_b, double *signalData_d, size_t signalLength, double *randomVector_d, const size_t randomLength,const size_t rate,
	int currentGen, double *bestAgent)
{
	extern __shared__ double agents_local[];
	size_t randomOffset = (gridDim.x*blockDim.x*currentGen*N+(blockIdx.x*N*gridDim.x+threadIdx.x*N))%randomLength;
	if(blockIdx.x==0&&threadIdx.x==0) printf("%lu/%lu\n",randomOffset,randomLength);

	if(blockIdx.x==0&&threadIdx.x==0)
		for (int i = 0; i < N+1; ++i) {
			agents_local[i] = bestAgent[i];
		}
	else
		for (int i = 0; i < N; ++i)
			agents_local[threadIdx.x*(N+1)+i] = randomVector_d[++randomOffset%randomLength];
	__syncthreads();
	double *child;
	child = (double *)malloc(sizeof(double)*N);
	if(currentGen!=0 && (blockIdx.x!=0||threadIdx.x!=0))
	{
		//  Reproduction
		double *parents[3];
		parents[0] = bestAgent;
		parents[1] = 0; parents[2] = 0;
		unsigned int randomParentIdx;
		for (int i = 1; i < 3; ++i)
		{
			do
			{
				randomParentIdx = floor(blockDim.x*randomVector_d[++randomOffset%randomLength]);
			} while(&agents_local[randomParentIdx*(N+1)] == parents[0] ||
				&agents_local[randomParentIdx*(N+1)] == parents[1] ||
				&agents_local[randomParentIdx*(N+1)] == parents[2]);
			parents[i] = &agents_local[randomParentIdx*(N+1)];
		}

		for (int i = 0; i < N; ++i) {
			double val = bestAgent[i];
			val += (F*(parents[1][i]-parents[2][i]));
			if(val<0.0) val=0.0;
			else if(val>1.0) val=1.0;
			child[i] = val;
		}
		//  Crossover
		unsigned int delta = floor(N*randomVector_d[++randomOffset%randomLength]);
		for (int i = 0; i < N; ++i)
			agents_local[threadIdx.x*(N+1)+i] =
				(delta != i && randomVector_d[++randomOffset%randomLength]>R)?child[i]:bestAgent[i];
	}
	double volts = voltsConstantMin+agents_local[threadIdx.x*(N+1)]*(voltsConstantMax - voltsConstantMin);
	double omega = omegaConstantMin+agents_local[threadIdx.x*(N+1)+1]*(omegaConstantMax - omegaConstantMin);
	double phi = agents_local[threadIdx.x*(N+1)+2]*phiConstant;
	double t,diff,accum=0.0;
	for(size_t pos = 0; pos < signalLength; pos++)
	{
		t=(double)pos/(double)rate;
		diff = volts*sin(
				omega*t+
				phi
			)-signalData_d[pos];
		accum += pow(diff,2);
	}
	agents_local[threadIdx.x*(N+1)+N] = accum;
	__syncthreads();	//Wait for all threads of the block to end
	//Calculate best agent
	if(threadIdx.x==0)
	{
		double *bestAgentOfBlock = &agents_local[0];
		for (int i = 1; i < blockDim.x; ++i)
			if(agents_local[i*(N+1)+N] < bestAgentOfBlock[N])
				bestAgentOfBlock = &agents_local[i*(N+1)];
		for (int i = 0; i < N+1; ++i)
			bestScoreAndParameters_b[blockIdx.x*(N+1)+i] = bestAgentOfBlock[i];
	}
	__syncthreads();	//Wait for all threads of the block to end
}
extern "C" void runDE(double *signalData, size_t signalLength, double *randomVector, const size_t randomLength,
	const unsigned long int S, const unsigned long int G, const double F, const double R, const size_t N, const double epsilon, const size_t rate)
{
	int nBlocks = (int)ceil((double)S/32);
	int nTpb = 32;
    double *signalData_d;
    HANDLE_ERROR( cudaMalloc((void **)&signalData_d, signalLength*sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy( signalData_d, signalData, signalLength*sizeof(double), cudaMemcpyHostToDevice) );
    double *randomVector_d;
    HANDLE_ERROR( cudaMalloc((void **)&randomVector_d, randomLength*sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy( randomVector_d, randomVector, randomLength*sizeof(double), cudaMemcpyHostToDevice) );
    double *bestScoreAndParameters_b;
    HANDLE_ERROR( cudaMallocManaged(&bestScoreAndParameters_b,nBlocks*(N+1)*sizeof(double)) );

    //Use max threads per block
    struct cudaDeviceProp prop;
    int cudaDevice;
    HANDLE_ERROR( cudaGetDevice(&cudaDevice) );
    HANDLE_ERROR( cudaGetDeviceProperties(&prop, cudaDevice) );
    printf("Device used: %s\nmaxThreadsPerBlock: %d\nmaxGridSize: %dx%dx%d\n",
    	prop.name, prop.maxThreadsPerBlock, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    //S is for population size (number of agents)
    //Use one agent per thread, Use 32 agents per block, rounded up
    double *bestAgent;
    HANDLE_ERROR( cudaMallocManaged(&bestAgent,(N+1)*sizeof(double)) );
    bestAgent[N]=DBL_MAX;
    for (int currentGen = 0; currentGen <= G; ++currentGen)
    {
    	if(currentGen == 28)
    		printf("Warning!\n");
    	printf("Generation: %d\n",currentGen);
		runDEagent<<< nBlocks, nTpb, nTpb*(N+1)*sizeof(double)>>>(S,G,F,R,N,bestScoreAndParameters_b,signalData_d,signalLength,randomVector_d,randomLength,rate,currentGen, bestAgent);
		cudaDeviceSynchronize(); //Wait for all blocks to finish
		//Get best of them from bestScoreAndParameters (serial mode)
		for (int b = 0; b < nBlocks; ++b)
			if(bestScoreAndParameters_b[b*(N+1)+N] < bestAgent[N])
				for (int i = 0; i < N+1; ++i) bestAgent[i] = bestScoreAndParameters_b[b*(N+1)+i];
		double volts = voltsConstantMin+bestAgent[0]*(voltsConstantMax - voltsConstantMin);
		double omega = omegaConstantMin+bestAgent[1]*(omegaConstantMax - omegaConstantMin);
		double phi = bestAgent[2]*phiConstant;
		printf("Best agent in generation %d. %lf, %lf, %lf, score: %lf\n",
			currentGen, volts, omega, phi, bestAgent[N]);
    }
    HANDLE_ERROR( cudaGetLastError() );

    HANDLE_ERROR( cudaFree(signalData_d) );
    HANDLE_ERROR( cudaFree(bestScoreAndParameters_b) );

    cudaDeviceReset();
	return;
}
