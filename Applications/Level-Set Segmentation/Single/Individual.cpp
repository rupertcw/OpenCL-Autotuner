#include "Individual.h"

using namespace std;

Individual::Individual()
{
    fitness = 999999.0f;
}

Individual::Individual(vector<int> params)
{
    values = params;
    fitness = 999999.0f;
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