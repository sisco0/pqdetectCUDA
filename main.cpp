#include <iostream>

#include <stdlib.h>

#include "signalGenerator/SignalObject.h"
#include "signalGenerator/SignalGenerator.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

#include <algorithm>
#include <random>
#include <functional>

extern "C" void runDE(double *signalData, size_t signalLength, double *randomVector, const size_t randomLength,
	const unsigned long int S, const unsigned long int G, const double F, const double R, const size_t N, const double epsilon, const size_t rate);

int main(void)
{
    const size_t rate = 60*20;
	const size_t signalLength = 2*rate; //Generate 3 seconds
	SignalGenerator::SineSignalGenerator gen1 = SignalGenerator::SineSignalGenerator();
	gen1.setRate(rate).setFrequency(60).setPhase(3.1415).setAmplitude(127*sqrt(2));
	SignalGenerator::WhiteNoiseSignalGenerator gen2 = SignalGenerator::WhiteNoiseSignalGenerator();
	gen2.setMean(0).setStandardDeviation(10);
	//Generate the sum of them
	SignalObject signal1 = gen1.generate(signalLength);
	SignalObject signal2 = gen2.generate(signalLength);
	SignalObject sumSignal = signal1 + signal2;
	sumSignal.addSagSwell(100, 199, 0.5);



	double signalSS = 0;
	for_each( sumSignal.m_data->begin(),sumSignal.m_data->end(), [&signalSS](double a) { signalSS += pow(a,2); });

	//Random initialization
	std::random_device rnd_device;
	std::mt19937 mersenne_engine(rnd_device());
	std::uniform_real_distribution<double> dist(0, 1);
    auto gen = std::bind(dist, mersenne_engine);
    const unsigned long int S = 32*100;	//Use a multiple of 32
    const unsigned long int G = 28;
    const double F = 1.4, R = 0.5;
    const size_t N = 3; //Number of parameters to estimate
    const double epsilon = 1e-10;
    const size_t randomVectorSize = S*(N+1)+10000;	//Its recommended a big number
    std::vector<double> randomVector(randomVectorSize);
    std::generate(begin(randomVector), end(randomVector), gen);

    runDE((double *)&(sumSignal.m_data[0]),sumSignal.m_data->size(),(double *)&(randomVector[0]),randomVectorSize,S,G,F,R,N,epsilon,rate);


	return 0;
}
