from random import randint, shuffle, random, uniform
from copy import deepcopy
from operator import itemgetter, attrgetter

class Individual(object):
	def __init__(self, params):
		self.values = params[:]
		self.counter = 0
		self.rank = 0
		self.dominated = []
		self.d = None
		self.results = []

	def __repr__(self):
		return '('+str(self.values)+', '+str(self.counter)+', '+str(self.rank)+', '+str(self.d)+')'

def obj1(x):
	for i in range(0, len(x.values)) :
		if x.values[i] >= 0 and x.values[i] <= 640 :
			return x.values[i]*2
		else :
			return float("inf")

def obj2(x):
	for i in range(0, len(x.values)) :
		if x.values[i] >= 0 and x.values[i] <= 480 :
			return x.values[i]-2
		else :
			return float("inf")

def initialisePopulation(nParams):
	x = []
	i = 1

	while i < 5 :
		params = []
		for j in range(0, nParams) :
			params.append(randint(1, 1000))
		x.append(Individual(params))
		i = i + 1

	shuffle(x)

	return x

def calculateObjectives(P, obj):
	for i in P :
		results = []
		for m in obj :
			if m == 0:
				results.append(obj1(i))
			elif m == 1 :
				results.append(obj2(i))
		i.results = results[:]

def dominates(p, q):
	for i in range(0, len(p.results)) :
		if p.results[i] > q.results[i] :
			return False
	return True

def fastNonDominatedSort(P):
	F = []
	F.append([])
	# print "1: " + str(P)
	for p in P :
		# print 'p: ' + str(p)
		p.dominated = []
		p.counter = 0
		p.rank = 0

		for q in P:
			# print 'q: ' + str(q)
			# print 'dominates ?: ' + str(dominates(p, q))
			if dominates(p, q) :
				p.dominated.append(q)
			elif dominates(q, p) :
				p.counter = p.counter + 1

		if p.counter == 0 :
			p.rank = 1
			F[0].append(p)

	i = 1
	# print "2: " + str(P)
	while len(F[i-1]) != 0 :
		# print i
		Q = []
		for p in F[i-1] :
			for q in p.dominated :
				# print 'before counter: ' + str(q.counter)
				q.counter = q.counter - 1
				# print 'after: ' + str(q)
				if q.counter == 0 :
					q.rank = i + 1
					Q.append(q)
		i = i + 1
		F.append(Q)
	return F

def objectiveSort(f, m):
	results = []

	for i in f:
		results.append([i.results[m], i])
	# results.append([3, Individual(2)])
	# print 'before sort: ' + str(results)
	results.sort()
	# print 'after sort: ' + str(results)
	results = [r[1] for r in results]
	# print 'after manipulation: ' + str(results)
	return results

def getMax(P, m):
	return max([ind.results[m] for ind in P])

def getMin(P, m):
	return min([ind.results[m] for ind in P])

def crowdedComparison(p, q):
	if (p.rank < q.rank) or (p.rank == q.rank and p.d > q.d):
		return True
	return False

def binaryTournamentSelection(P, co):
	parent1 = P[randint(0, len(P)-1)]
	parent2 = P[randint(0, len(P)-1)]
	
	if co == "better" :
		if (parent1.d is not None) and (parent1.rank == parent2.rank) :
			if parent1.d > parent2.d :
				return Individual(parent1.values)
			else :
				return Individual(parent2.values)
		elif parent1.rank < parent2.rank :
			return Individual(parent1.values)
		return Individual(parent2.values)
	elif co == "crowded" :
		if crowdedComparison(parent1, parent2) :
			return Individual(parent1.values)
		else :
			return Individual(parent2.values)

def simulatedBinaryCrossover(p1, p2):
	nc = 20
	c1 = []
	c2 = []

	# cross
	for k in range(0, len(p1.values)) :
		u = uniform(0, 1)
		# print 'u: ' + str(u)

		if u <= 0.5 :
			B = (2*u)**(1/float((nc+1)))
		else :
			B = 1/float(float(2*(1-u))**(1/float(nc+1)))

		# print 'B: ' + str(B)

		v1 = 0.5*((1-B)*p1.values[k] + (1+B)*p2.values[k])
		v2 = 0.5*((1+B)*p1.values[k] + (1-B)*p2.values[k])
		# print 'v1: ' + str(v1)
		# print 'v2: ' + str(v2)

		c1.append(int(v1))
		c2.append(int(v2))

	return Individual(c1), Individual(c2)

def singlePointCrossover(p1, p2):
	cut = int(round(len(p1)/2))
	# p1Bits = "{0:b}".format(p1.value)
	# p2Bits = "{0:b}".format(p2.value)

	# if len(p1Bits) < len(p2Bits) :
	# 	p1Bits.zfill(len(p2Bits))
	# elif len(p1Bits) > len(p2Bits) :
	# 	p2Bits.zfill(len(p1Bits))

	# c1Bits = "".zfill(len(p1Bits))
	# c2Bits = "".zfill(len(p1Bits))

	# cross

	# return as Individual
	return c1, c2

def polynomialMutation(c, p, pu, pl):
	nm = 20

	#mutate
	for k in range(0, len(p.values)) :
		r = uniform(0, 1)

		if r < 0.5 :
			delta = ((2*r)**(1/float((nm + 1)))) - 1
		else :
			delta = 1 - ((2*(1 - r))**(1/float((nm + 1))))

		# print 'Delta: ' + str(delta)

		c.values[k] = int(p.values[k] + (pu[k]-pl[k])*delta)

def bitwiseMutation(c, rate):
	# if random() < rate :
		# cBits = "{0:b}".format(c.value)

		#mutate
		# for k in range(0, len(cBits)) :

		# return as Individual
	return c

def newGeneration(parents, xRate, mRate, pu, pl):
	G = []

	for i in range(0, len(Q), 2) :
		# Crossover
		if uniform(0, 1) < xRate :
			child1, child2 = simulatedBinaryCrossover(parents[i], parents[i+1])
			# print 'Child1: ' + str(child1)
			# print 'Child2: ' + str(child2)
		else :
			child1 = Individual(parents[i].values)
			child2 = Individual(parents[i+1].values)
			# print 'Child1 - no X: ' + str(child1)
			# print 'Child2 - no X: ' + str(child2)

		# Mutation
		if uniform(0, 1) < mRate :
			polynomialMutation(child1, parents[i], pu, pl)
			# print 'Child1 after Mutation: ' + str(child1)

		if uniform(0, 1) < mRate :	
			polynomialMutation(child2, parents[i+1], pu, pl)
			# print 'Child2 after Mutation: ' + str(child2)

		G.append(child1)
		G.append(child2)
	return G

def calculateCrowdingDistance(P, F, obj):
	for f in F :
		# print 'front: ' + str(f)
		if len(f) != 0 :
			n = len(f)
			for ind in f :
				ind.d = 0
			for m in obj :
				# print 'before: ' + str(f)
				f = list(objectiveSort(f, m))

				f[0].d = float("inf")
				f[-1].d = float("inf")

				# print 'after: ' + str(f)
				# print 'here'
				maxN = getMax(P, m)
				minN = getMin(P, m)
				rge = maxN - minN

				if rge != 0 :
					for k in range(1, n-1) :
						v = f[k].d + (f[k+1].results[m] - f[k-1].results[m]) / rge
						f[k].d = v
						# print '1: ' + str(f[k].d)
						# print 'a: ' + str(f[k+1].results[m])
						# print 'b: ' + str(f[k-1].results[m])
						# print '2: ' + str(f[k+1].results[m] - f[k-1].results[m])
						# print '3: ' + str(rge)

def selectNewPopulation(P, F, obj):
	children = []
	lastFront = 0
	size = 0

	calculateCrowdingDistance(P, F, obj)

	for f in F:
		if (len(children) + len(f)) > (len(P)/2) :
			break
		for ind in f :
			children.append(ind)
		lastFront = lastFront + 1

	# print 'children: ' + str(children)

	if (len(P)/2)-len(children) > 0 :
		desc = []

		if len(F[lastFront]) > 1 :
			# cc sort
			for i in range(0, len(F[lastFront])-1) :
				if crowdedComparison(F[lastFront][i], F[lastFront][i+1]) :
					desc.append(F[lastFront][i])
					desc.append(F[lastFront][i+1])
				else :
					desc.append(F[lastFront][i+1])
					desc.append(F[lastFront][i])

			desc.reverse()
			F[lastFront] = desc[:]
			# print 'After desc sort: ' + str(F[lastFront])

		for j in range(0, (len(P)/2)-len(children)) :
			children.append(F[lastFront][j])

	return children

def selectParents(Q, P, N, co):
	while len(Q) != N :
		parent = binaryTournamentSelection(P, co)
		Q.append(parent)

# pu = [640, 480, 640, 480]
# pl = [0, 0, 0, 0]
N = 4
pu = [100]
pl = [0]
xRate = 0.9
mRate = 0.3
nParams = 1
generations = 1
obj = [0, 1]
Q = []

P = initialisePopulation(nParams)
# print 'Initial Population: ' + str(P)

calculateObjectives(P, obj)
# for i in P :
# 	print 'Initial Results: ' + str(i.results)

# for i in P :
# 	print i
# 	for j in i.dominated :
# 		print j
# 	print ''

F = fastNonDominatedSort(P)
# print 'Fronts: ' + str(F)

# for i in P :
# 	print i
# 	for j in i.dominated :
# 		print j
# 	print ''

selectParents(Q, P, N, "better")
print 'Selected Parents: ' + str(Q)

G = newGeneration(Q, xRate, mRate, pu, pl)
# print 'New Generation: ' + str(G)

calculateObjectives(G, obj)

for i in range(0, generations) :
	Q = []
	# print i
	# print 'start1 - P should be 4: ' + str(len(P))
	P.extend(G)
	# print 'start2 - P should be 8: ' + str(len(P))
	# print '2N: ' + str(P)

	F = fastNonDominatedSort(P)
	# print 'New Fronts: ' + str(F)

	Q = selectNewPopulation(P, F, obj)
	print 'Selected Population: ' + str(Q)
	# print 'middle1 - P should be 8: ' + str(len(P))

	selectParents(Q, P, N, "crowded")
	# print 'Selected Parents: ' + str(Q)

	# print 'middle2 - P should be 8: ' + str(len(P))
	P = G[:]
	# print 'middle3 - P should be 4: ' + str(len(P))
	G = newGeneration(Q, xRate, mRate, pu, pl)

	calculateObjectives(G, obj)
	# print 'end - P should be 4: ' + str(len(P))


P.extend(G)
F = fastNonDominatedSort(P)
parents = selectNewPopulation(P, F, obj)

print 'Final Population: ' + str(parents)










