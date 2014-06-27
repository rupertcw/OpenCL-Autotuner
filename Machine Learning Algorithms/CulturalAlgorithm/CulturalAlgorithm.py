(from random import randint, shuffle, random, uniform
from copy import deepcopy
from operator import itemgetter, attrgetter

class Individual(object):
	def __init__(self, params, fitness=None):
		self.values = params[:]
		self.fitness = fitness

	def __repr__(self):
		return '('+str(self.values)+', '+str(self.fitness)+')'

def obj1(w, x, y, z):
	# return (w*2 - x + y) + z
	return x*2

	# return float("inf")

def initialisePopulation(searchSpace, N, nParams):
	x = []
	i = 0

	while i < N :
		params = []
		for j in range(0, nParams):
			params.append(randint(searchSpace[j][0], searchSpace[j][1]))

		x.append(Individual(params))
		i = i + 1

	shuffle(x)

	return x

def initialiseBeliefSpace(searchSpace):
	beliefSpace = {'Situational' : None}

	S = []
	for bounds in searchSpace :
		S.append(bounds)

	beliefSpace['Normative'] = S
	# print beliefSpace['Normative']

	return beliefSpace

def evaluatePopulation(P):
	for ind in P :
		ind.fitness = obj1(ind.values[0], ind.values[1], ind.values[2], ind.values[3])

def sortPopulation(P, rev=False):
	one = []
	for ind in P :
		two = []
		two.append(ind.values)
		two.append(ind.fitness)
		one.append(two[:])

	one = sorted(one, key=itemgetter(1), reverse=rev)

	for i in range(0, len(P)) :
		P[i] = Individual(one[i][0], one[i][1])

def updateSituational(beliefSpace, best):
	currBest = beliefSpace['Situational']

	if currBest is None or best.fitness < currBest.fitness :
		beliefSpace['Situational'] = best

def randInBounds(min, max):
	return min + ((max - min)*uniform(0, 1))

def mutate(individual, beliefSpace, searchSpace):
	params = []
	for i in range(0, len(individual.values)) :
		value = randInBounds(beliefSpace['Normative'][i][0], beliefSpace['Normative'][i][1])
		if value < searchSpace[i][0] :
			value = searchSpace[i][0]
		if value > searchSpace[i][1] :
			value = searchSpace[i][1]
		params.append(int(value))

	return params

def newGeneration(P, beliefSpace, searchSpace):
	G = []
	for ind in P :
		# print ind
		G.append(Individual(mutate(ind, beliefSpace, searchSpace)))

	return G

def binaryTournament(P):
	r1 = randint(0, len(P)-1)
	r2 = randint(0, len(P)-1)

	while r2 == r1 :
		r2 = randint(0, len(P)-1)

	# print 'r1, r2: ' + str(r1) + ', ' + str(r2)

	if P[r1].fitness < P[r2].fitness :
		return P[r1]

	return P[r2]

def selectNewPopulation(P, G):
	newPop = []
	i = 0
	P.extend(G)
	# print '2N: ' + str(P)
	while i < (len(P)/2) :
		newPop.append(binaryTournament(P))
		i = i + 1

	return newPop

def updateNormative(beliefSpace, accepted, nParams):
	for i in range(0, nParams) :
		beliefSpace['Normative'][i][0] = min([ind.values[i] for ind in accepted])
		beliefSpace['Normative'][i][1] = max([ind.values[i] for ind in accepted])

generations = 1
N = 8
nParams = 4
numAccepted = int(round(N*0.2))
searchSpace = [[0, 640], [0, 640], [0, 480], [0, 480]]

P = initialisePopulation(searchSpace, N, nParams)
print 'Initial Population: ' + str(P)

beliefSpace = initialiseBeliefSpace(searchSpace)
print 'Belief Space: ' + str(beliefSpace)

evaluatePopulation(P)
print 'P After Evaluation: ' + str(P)

sortPopulation(P)
print 'After Sort: ' + str(P)

best = P[0]
print 'best: ' + str(best)

updateSituational(beliefSpace, best)
print 'Updated Situational: ' + str(beliefSpace['Situational'])

for i in range(0, generations) :

	G = newGeneration(P, beliefSpace, searchSpace)
	print 'New Generation: ' + str(G)

	evaluatePopulation(G)
	print 'G After Evaluation: ' + str(G)

	sortPopulation(G)
	print 'G After Sort: ' + str(G)

	best = G[0]
	print 'G best: ' + str(best)

	updateSituational(beliefSpace, best)
	print 'Updated Situational: ' + str(beliefSpace['Situational'])

	P = selectNewPopulation(P, G)
	print 'New Population: ' + str(P)

	sortPopulation(P, True)
	print 'New Population After Inv Sort: ' + str(P)

	accepted = P[0:numAccepted]
	print 'Accepted Individuals: ' + str(accepted)

	updateNormative(beliefSpace, accepted, nParams)
	print 'Updated Normative: ' + str(beliefSpace["Normative"])

print 'Best Solution: ' + str(beliefSpace['Situational'])

