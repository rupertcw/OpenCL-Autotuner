#include <math.h>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <sstream>
#include <string>

using namespace std;

const int N = 8;
const int nGenerations = 2;
const int nParams = 4;

default_random_engine generator;
uniform_real_distribution<float> distribution(0.0, 1.0);

class Individual
{
	public:
		Individual();
		Individual(vector<int>);

		vector<int> values;
		int fitness;
		friend ostream& operator<< (ostream& o, Individual const& individual);
};

Individual::Individual()
{
	fitness = 999999;
}

Individual::Individual(vector<int> params)
{
	values = params;
	fitness = 999999;
}

ostream& operator<< (ostream& o, Individual const& individual)
{
	stringstream repr;
	repr.str(string());

	repr << "([";

	if (individual.values.size() > 1) {
		for (int i=0; i<individual.values.size()-1; i++)
		{
			repr << individual.values[i];
			repr << ", ";
		}
		repr << individual.values.back();
	} else {
		repr << individual.values.front();
	}

	repr << "], ";
	repr << individual.fitness << ")";

 	return o << repr.str();
}

void printPopulation(vector<Individual> P)
{
	for (int i=0; i<P.size(); i++)
		cout << "Individual " << i << ": " << P[i] << endl;
}

void printNormative(vector<vector<int> > normative)
{
	for (int i=0; i<normative.size(); i++)
		cout << "[" << normative[i][0] << ", " << normative[i][1] << "]" << endl;
}

int randint(int min, int max) { return rand() % max + min; }

// NOT RANDOM ENOUGH
float uniform() { return distribution(generator); }

void extend(vector<Individual>& v1, vector<Individual> v2)
{
	for (int i = 0; i < v2.size(); i++)
		v1.push_back(v2[i]);
}

bool fitnessComparison(Individual p, Individual q)
{
	if (p.fitness < q.fitness)
		return true;
	else
		return false;
}

int obj1(int w, int x, int y, int z) { return x*2; }

void initialisePopulation(vector<Individual>& P, vector<vector<int> > searchSpace)
{
	int i = 0;

	while (i < N)
	{
		vector<int> params(nParams);

		for (int j=0; j<params.size(); j++)
			params[j] = randint(searchSpace[j][0], searchSpace[j][1]);

		P.push_back(Individual(params));
		i = i + 1;
	}

	random_shuffle(P.begin(), P.end());
}

void initialiseBeliefSpace(vector<vector<int> > searchSpace, Individual& situational, vector<vector<int> >& normative)
{
	situational = Individual();
	normative = searchSpace;
}

void evaluatePopulation(vector<Individual>& P)
{
	for (int i=0; i<P.size(); i++)
		P[i].fitness = obj1(P[i].values[0], P[i].values[1], P[i].values[2], P[i].values[3]);
}

void sortPopulation(vector<Individual>& P, int co)
{
	sort(P.begin(), P.end(), fitnessComparison);

	if (co == 1)
		reverse(P.begin(), P.end());
}

void updateSituational(Individual& situational, Individual best)
{
	if ((situational.values.empty()) || (best.fitness < situational.fitness))
		situational = best;
}

float randInBounds(int min, int max)
{
	return min + ((max - min)* uniform());
}

void mutate(Individual individual, vector<int>& params, vector<vector<int> > normative, vector<vector<int> > searchSpace)
{
	params.clear();

	for (int i=0; i<individual.values.size(); i++)
	{
		float value = randInBounds(normative[i][0], normative[i][1]);

		if (value < searchSpace[i][0])
			value = searchSpace[i][0];
		if (value > searchSpace[i][1])
			value = searchSpace[i][1];

		params.push_back((int)value);
	}
}

void newGeneration(vector<Individual>& P, vector<Individual>& G, vector<vector<int> > normative, vector<vector<int> > searchSpace)
{
	vector<int> params;

	G.clear();

	for (int i=0; i<P.size(); i++){
		mutate(P[i], params, normative, searchSpace);

		G.push_back(Individual(params));
	}
}

Individual binaryTournament(vector<Individual> P)
{
	int r1 = randint(0, P.size()-1);
	int r2 = randint(0, P.size()-1);

	while (r2 == r1)
		r2 = randint(0, P.size()-1);

	if (P[r1].fitness < P[r2].fitness)
		return P[r1];
	else
		return P[r2];
}

void selectNewPopulation(vector<Individual>& P, vector<Individual>& P2, vector<Individual> G)
{
	int i = 0;

	P2.clear();
	extend(P, G);

	while (i < (P.size()/2))
	{
		P2.push_back(binaryTournament(P));
		i = i + 1;
	}
}

void updateNormative(vector<vector<int> >& normative, vector<Individual> accepted)
{
	for (int i=0; i<nParams; i++)
	{
		vector<int> params;

		for (int j=0; j<accepted.size(); j++)
			params.push_back(accepted[j].values[i]);

		sort(params.begin(), params.end());

		normative[i][0] = params.front();
		normative[i][1] = params.back();
	}
}

int main()
{
	srand(time(0));

	int numAccepted = round(N*0.2);

	Individual situational;

	vector<Individual> P;
	vector<Individual> P2;
	vector<Individual> G;
	vector<Individual> accepted;

	vector<int> ss1(2);
	vector<int> ss2(2);
	vector<int> ss3(2);
	vector<int> ss4(2);
	vector<vector<int> > searchSpace(4);
	vector<vector<int> > normative;

	ss1[0] = 0;
	ss1[1] = 640;
	ss2[0] = 0;
	ss2[1] = 640;
	ss3[0] = 0;
	ss3[1] = 480;
	ss4[0] = 0;
	ss4[1] = 480;
	searchSpace[0] = ss1;
	searchSpace[1] = ss2;
	searchSpace[2] = ss3;
	searchSpace[3] = ss4;

	printNormative(searchSpace);

	initialisePopulation(P, searchSpace);
	printPopulation(P);

	initialiseBeliefSpace(searchSpace, situational, normative);
	printNormative(normative);

	evaluatePopulation(P);
	printPopulation(P);

	sortPopulation(P, 0);
	printPopulation(P);

	Individual best = P[0];
	cout << "Best: " << best << endl;

	updateSituational(situational, best);
	cout << "Updated situational: " << situational << endl;

	// for (int i = 0; i < nGenerations; i++)
	// {
	newGeneration(P, G, normative, searchSpace);
	cout << "New Generation" << endl;
	printPopulation(G);

	evaluatePopulation(G);
	cout << "After Evaluation" << endl;
	printPopulation(G);

	sortPopulation(G, 0);
	cout << "After Sort" << endl;
	printPopulation(G);

	best = G[0];
	cout << "Best: " << best << endl;

	updateSituational(situational, best);
	cout << "Updated situational: " << situational << endl;

	selectNewPopulation(P, P2, G);
	cout << "New Population" << endl;
	P = P2;
	printPopulation(P);

	sortPopulation(P, 1);
	cout << "After Sort" << endl;
	printPopulation(P);

	accepted.clear();
	for (int i = 0; i < numAccepted; i++)
		accepted.push_back(P[i]);
	cout << "Accepted Individuals" << endl;
	printPopulation(accepted);

	updateNormative(normative, accepted);
	cout << "Updated Normative" << endl;
	printNormative(normative);
	// }

	cout << "Best Solution: " << situational << endl;
}









