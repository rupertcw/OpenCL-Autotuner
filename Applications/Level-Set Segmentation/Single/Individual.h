#include <string.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <sstream>
#include <string>

using namespace std;

class Individual
{
    public:
        Individual();
        Individual(vector<int>);

        vector<int> values;
        double fitness;
        friend ostream& operator<< (ostream& o, Individual const& individual);
};