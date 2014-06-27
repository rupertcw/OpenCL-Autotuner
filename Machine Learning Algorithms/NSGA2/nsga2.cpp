#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <ctime>
#include <random>
#include <limits>

using namespace std;

const int nParams = 1;
const int nFirst = 4;
const int nObjs = 2;
const int nGenerations = 1;
const int nc = 20;
const int nm = 20;

default_random_engine generator;
uniform_real_distribution<float> distribution(0.0, 1.0);

class Individual
{
	public:
		Individual();
		Individual(vector<int>);

		vector<int> values;
		vector<Individual*> dominated;
		int counter, rank;
		vector<float> results;
		float d;
		friend ostream& operator<< (ostream& o, Individual const& individual);
};

Individual::Individual()
{
	counter = 0;
	rank = 0;
	d = 0;
}

Individual::Individual(vector<int> params)
{
	values = params;
	counter = 0;
	rank = 0;
	d = 999999;
	results = vector<float>(nObjs, 0);
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

	repr << "], [";

	if (individual.results.size() > 1) {
		for (int i=0; i<individual.results.size()-1; i++)
		{
			repr << individual.results[i];
			repr << ", ";
		}
		repr << individual.results.back();
	} else {
		repr << individual.results.front();
	}
	
	repr << "], ";
	repr << individual.counter << ", ";
	repr << individual.rank << ", ";
	repr << individual.d << ")";

 	return o << repr.str();
}

void printPopulation(vector<Individual> P)
{
	for (int i=0; i<P.size(); i++)
		cout << "Element " << i << ": " << P[i] << endl;
}

void printDominated(Individual I)
{
	for (int i=0; i<I.dominated.size(); i++)
		cout << "Dominated element counter: " << I.dominated[i]->counter << endl;
}

void printFronts(vector<vector<Individual> > F)
{
	for (int i=0; i<F.size(); i++)
		for (int j=0; j<F[i].size(); j++)
			cout << "Front " << F[i][j].rank << ": " << F[i][j] << endl;
}

int randint(int min, int max) { return rand() % max + min; }

// NOT RANDOM ENOUGH
float uniform() { return distribution(generator); }

bool crowdedComparison(Individual p, Individual q)
{
	if ((p.rank < q.rank) || ((p.rank == q.rank) && (p.d > q.d)))
		return true;
	return false;
}

bool tupleComparison(tuple<float, Individual> t1, tuple<float, Individual> t2)
{
	if (get<0>(t1) < get<0>(t2))
		return true;
	else
		return false;
}

bool maxComparison(Individual ind, int m, float max)
{
	if (ind.results[m] > max)
		return true;
	return false;
}

bool minComparison(Individual ind, int m, float min)
{
	if (ind.results[m] < min)
		return true;
	return false;
}

void extend(vector<Individual>& v1, vector<Individual> v2)
{
	for (int i = 0; i < v2.size(); i++)
		v1.push_back(v2[i]);
}

int obj1(Individual x)
{
	for (int i = 0; i < x.values.size(); i++)
	{
		if ((x.values[i] >= 0) && (x.values[i] <= 640))
			return x.values[i]*2;
	}

	return numeric_limits<int>::max();
}

int obj2(Individual x)
{
	for (int i = 0; i < x.values.size(); i++)
	{
		if ((x.values[i] >= 0) && (x.values[i] <= 480))
			return x.values[i]-2;
	}

	return numeric_limits<int>::max();
}

void initialisePopulation(vector<Individual>& P)
{
	int i = 0;

	while (i < nFirst)
	{
		vector<int> params(nParams);

		for (int j=0; j<params.size(); j++)
			params[j] = randint(1, 100);

		P[i] = Individual(params);
		i = i + 1;
	}

	random_shuffle(P.begin(), P.end());
}

void calculateObjectives(vector<Individual>& P, vector<int> objs)
{
	for (int i = 0; i < P.size(); i++)
	{
		for (int j = 0; j < objs.size(); j++)
		{
			if (objs[j] == 0) {
				// cout << "objs j: " << objs[j] << endl;
				// cout << "Before i, j, res: " << i << ", " << j << ", " << P[i].results[j] << endl;
				// cout << "Assumed result: " << obj1(P[i]) << endl;
				P[i].results[j] = obj1(P[i]);
				// cout << "After i, j, res: " << i << ", " << j << ", " << P[i].results[j] << endl;
			} else if (objs[j] == 1) {
				P[i].results[j] = obj2(P[i]);
			}
		}
	}
}

bool dominates(Individual p, Individual q)
{
	for (int i = 0; i < p.results.size(); i++)
		if (p.results[i] > q.results[i])
			return false;
	return true;
}

void fastNonDominatedSort(vector<Individual>& P, vector<vector<Individual> >& F)
{
	vector<Individual> S;
	F.clear();

	// cout << "Before" << endl;
	// printFronts(F);
	// cout << "After" << endl;

	for (int i = 0; i < P.size(); i++)
	{
		P[i].counter = 0;
		P[i].rank = 0;
		P[i].dominated.clear();

		for (int j = 0; j < P.size(); j++)
		{
			// cout << "rank: " << P[i].rank << endl;
			if (dominates(P[i], P[j])) {
				// cout << P[j].counter << endl;
				P[i].dominated.push_back(&P[j]);
				// printDominated(P[i]);
				// cout << "here1" << endl;
			} else if (dominates(P[j], P[i])) {
				// cout << "before counter: " << P[i].counter << endl;
				P[i].counter = P[i].counter + 1;
				// cout << "after counter: " << P[i].counter << endl;
			}
		}

		if (P[i].counter == 0) {
			P[i].rank = 1;
			// cout << "here2" << endl;
			// printDominated(P[i]);
			S.push_back(P[i]);
			// printDominated(F[0][i]);
			// cout << "here3" << endl;
		}
	}

	F.push_back(S);
	int i = 1;
	// cout << "here" << endl;
	// printFronts(F);

	while (F[i-1].size() != 0)
	{
		vector<Individual> Q;
		// cout << "i: " << i-1 << endl;
		// cout << "size of F[i-1]: " << F[i-1].size() << endl;

		for (int j = 0; j < F[i-1].size(); j++)
		{
			// cout << "i: " << i-1 << ", j: " << j << endl;
			// cout << "size of F[i-1] dominated: " << F[i-1][j].dominated.size() << endl;
			for (int k = 0; k < F[i-1][j].dominated.size(); k++)
			{
				// cout << "Individual: " << *F[i-1][j].dominated[k] << endl;
				// cout << "counter1: " << F[i-1][j].dominated[k]->counter << endl;
				F[i-1][j].dominated[k]->counter = F[i-1][j].dominated[k]->counter - 1;
				// cout << "counter2: " << F[i-1][j].dominated[k]->counter << endl;

				if (F[i-1][j].dominated[k]->counter == 0) {

					F[i-1][j].dominated[k]->rank = i + 1;
					// cout << "here" << endl;
					// Individual ind = *(F[i-1][j]->dominated[k]);
					// cout << ind << endl;
					Q.push_back(*F[i-1][j].dominated[k]);
				}
			}
		}

		i = i + 1;
		F.push_back(Q);
	}

	// cout << "hereSeg" << endl;
	F.resize(F.size()-1);
	// cout << "F Size: " << F.size() << endl;
}

Individual binaryTournamentSelection(vector<Individual> P, int co)
{
	Individual parent1 = P[randint(0, P.size()-1)];
	Individual parent2 = P[randint(0, P.size()-1)];

	if (co == 0) {
		if ((parent1.d != 9999) && (parent1.rank == parent2.rank)) {
			if (parent1.d > parent2.d)
				return parent1;
			else
				return parent2;
		} else if (parent1.rank < parent2.rank)
			return parent1;
		return parent2;
	} else if (co == 1) {
		if (crowdedComparison(parent1, parent2))
			return parent1;
		else
			return parent2;
	} else
		return parent1;
}

void selectNewParents(vector<Individual>& Q, vector<Individual> P, int co)
{
	Q.clear();

	while (Q.size() != nFirst)
		Q.push_back(binaryTournamentSelection(P, co));
}

void simulatedBinaryCrossover(Individual p1, Individual p2, vector<Individual>& children)
{
	vector<int> c1;
	vector<int> c2;

	for (int k = 0; k < p1.values.size(); k++)
	{
		float u = uniform();
		float B = 0.0f;

		if (u <= 0.5f) {
			B = pow((2*u), (1.0f/(nc + 1)));
		}
		else {
			B = 1.0f/(pow((2*(1.0f - u)), (1.0f/(nc + 1))));
		}

		cout << "u: " << u << endl;
		// cout << "B: " << B << endl;

		float v1 = 0.5f*(((1.0f - B)*p1.values[k]) + ((1.0f + B)*p2.values[k]));
		float v2 = 0.5f*(((1.0f + B)*p1.values[k]) + ((1.0f - B)*p2.values[k]));

		// cout << "v1, (int)v1: " << v1 << ", " << (int)v1 << endl;
		// cout << "v2, (int)v2: " << v2 << ", " << (int)v2 << endl;

		c1.push_back((int)v1);
		c2.push_back((int)v2);
	}

	children.push_back(Individual(c1));
	children.push_back(Individual(c2));
}

void polynomialMutation(Individual& c, Individual p, vector<int> pl, vector<int> pu)
{
	for (int k = 0; k < p.values.size(); k++)
	{
		float r = uniform();
		float delta = 0.0f;

		// cout << "r: " << r << endl;

		if (r < 0.5f)
			delta = (pow((2*r), (1.0f/(nm + 1)))) - 1.0f;
		else{
			delta = 1.0f - (pow((2*(1.0f - r)), (1.0f/(nm + 1))));

			// cout << "delta: " << delta << endl;
		}

		// cout << "pu - pl * delta: " << ((pu[k] - pl[k])*delta) << endl;

		c.values[k] = (int)(p.values[k] + ((pu[k] - pl[k])*delta));
	}
}

void newGeneration(vector<Individual> parents, vector<Individual>& G, float xRate, float mRate, vector<int> pl, vector<int> pu)
{
	G.clear();

	for (int i = 0; i < nFirst; i+=2)
	{
		Individual child1, child2;
		vector<Individual> children;

		// Crossover
		if (uniform() < xRate) {
			// cout << "crossover" << endl;
			simulatedBinaryCrossover(parents[i], parents[i+1], children);
			child1 = children[0];
			child2 = children[1];
		} else {
			// cout << "no crossover" << endl;
			child1 = Individual(parents[i].values);
			child2 = Individual(parents[i+1].values);
		}

		// cout << "Before c1: " << child1 << endl;
		// cout << "Before c2: " << child2 << endl;

		// // Mutation
		if (uniform() < mRate)
			polynomialMutation(child1, parents[i], pl, pu);

		if (uniform() < mRate)
			polynomialMutation(child2, parents[i+1], pl, pu);

		// cout << "After c1: " << child1 << endl;
		// cout << "After c2: " << child2 << endl;

		G.push_back(child1);
		G.push_back(child2);
	}
}

// Might have to redo in normal C++
void objectiveSort(vector<Individual>& f, int m)
{
	vector<tuple<float, Individual> > results;

	for (int i = 0; i < f.size(); i++)
		results.push_back(make_tuple(f[i].results[m], f[i]));

	sort(results.begin(), results.end(), tupleComparison);

	for (int i = 0; i < results.size(); i++)
		f[i] = get<1>(results[i]);
}

float getMax(vector<Individual> P, int m)
{
	float max = 0;

	for (int i = 0; i < P.size(); i++)
		if (P[i].results[m] > max)
			max = P[i].results[m];
	return max;

	// return max_element(P.begin(), P.end(), maxComparison);
}

float getMin(vector<Individual> P, int m)
{
	float min = 999999;

	for (int i = 0; i < P.size(); i++)
		if (P[i].results[m] < min)
			min = P[i].results[m];
	return min;

	// return min_element(P.begin(), P.end(), minComparison);
}

// Sorting on same numbers useless!! => F[i][k + 1].results[objs[j]] - F[i][k - 1].results[objs[j]] = 0 ALWAYS
void calculateCrowdingDistance(vector<Individual> P, vector<vector<Individual> >& F, vector<int> objs)
{
	for (int i = 0; i < F.size(); ++i)
	{
		// cout << "here2" << endl;
		if (F[i].size() != 0)
		{
			// cout << "here3" << endl;
			int n = F[i].size();

			for (int j = 0; j < F[i].size(); j++)
				F[i][j].d = 0;

			for (int j = 0; j < objs.size(); j++)
			{
				objectiveSort(F[i], objs[j]);

				F[i][0].d = 0;
				F[i].back().d = numeric_limits<int>::max();

				float maxN = getMax(P, objs[j]);
				float minN = getMin(P, objs[j]);
				float rge = maxN - minN;

				if (rge != 0)
					for (int k = 1; k < n-1; k++)
						F[i][k].d = F[i][k].d + (F[i][k + 1].results[objs[j]] - F[i][k - 1].results[objs[j]])/rge;
			}
		}
	}
}

void selectNewPopulation(vector<Individual> P, vector<vector<Individual> >& F, vector<int> objs, vector<Individual>& R)
{
	int lastFront = 0, size = 0;
	R.clear();

	calculateCrowdingDistance(P, F, objs);

	for (int i = 0; i < F.size(); i++)
	{
		if ((R.size() + F[i].size()) > nFirst)
			break;

		for (int j = 0; j < F[i].size(); j++)
			R.push_back(F[i][j]);

		lastFront = lastFront + 1;
	}

	cout << "New Population - Size of R: " << R.size() << endl;

	if ((nFirst - R.size()) > 0)
	{
		vector<Individual> desc;

		if (F[lastFront].size() > 1)
		{
			for (int i = 0; i < F[lastFront].size() - 1; i++)
			{
				if (crowdedComparison(F[lastFront][i], F[lastFront][i + 1])) {
					desc.push_back(F[lastFront][i]);
					desc.push_back(F[lastFront][i + 1]);
				} else {
					desc.push_back(F[lastFront][i + 1]);
					desc.push_back(F[lastFront][i]);
				}
			}
			reverse(desc.begin(), desc.end());
			F[lastFront] = desc;
		}

		for (int i = 0; i < (nFirst - R.size()); i++)
			R.push_back(F[lastFront][i]);
	}
}

int main()
{
	float xRate = 0.9, mRate = 0.3;

	vector<int> pl(1, 0);
	vector<int> pu(1, 100);
	vector<int> objs(2);
	objs[0] = 0;
	objs[1] = 1;

	vector<Individual> P(nFirst);
	vector<Individual> Q;
	vector<Individual> G;
	vector<Individual> R;

	vector<vector<Individual> > F;

	srand(time(0));

	initialisePopulation(P);
	cout << "Initial Population:" << endl;
	printPopulation(P);

	calculateObjectives(P, objs);
	// cout << "After calculate Objectives:" << endl;
	// printPopulation(P);

	fastNonDominatedSort(P, F);
	cout << "Non Dominated Sorting:" << endl;
	printFronts(F);

	selectNewParents(Q, P, 0);
	cout << "New Selected Parents:" << endl;
	printPopulation(Q);

	newGeneration(Q, G, xRate, mRate, pl, pu);
	cout << "New Generation:" << endl;
	printPopulation(G);

	calculateObjectives(G, objs);
	cout << "After calculate Objectives2:" << endl;
	printPopulation(G);

	for (int i = 0; i < nGenerations; i++)
	{
		extend(P, G);
		cout << "After extension:" << endl;
		printPopulation(P);

		fastNonDominatedSort(P, F);
		cout << "Non Dominated Sorting:" << endl;
		printFronts(F);

		selectNewPopulation(P, F, objs, R);
		cout << "New Population Selected:" << endl;
		printPopulation(R);

		selectNewParents(R, P, 1);
		cout << "New Selected Parents:" << endl;
		printPopulation(R);

		P.clear();
		P = G;

		newGeneration(R, G, xRate, mRate, pl, pu);
		cout << "New Generation:" << endl;
		printPopulation(G);

		calculateObjectives(G, objs);
		cout << "After calculate ObjectivesN:" << endl;
		printPopulation(G);
	}

	extend(P, G);
	cout << "After extension:" << endl;
	printPopulation(P);

	fastNonDominatedSort(P, F);
	cout << "Non Dominated Sorting:" << endl;
	printFronts(F);

	selectNewPopulation(P, F, objs, R);
	cout << "New Population Selected:" << endl;
	printPopulation(R);
}









