import numpy as np; 
import random as rd;
import operator; 
import pandas as pd; 
from matplotlib import pyplot as plt; 
import random as rd;
import collections;
from itertools import permutations;


class Graph():
    """
    Graph: class creates distance matrix for TSP problem. 
    """
    size= 0; 
    def __init__(self, rank: int, matrix: list = None):
        """
            :param rank: rank of the cost matrix 
        """
        self.rank = rank
        #print(list)
        if matrix is not None: 
            self.matrix = np.asarray(matrix);
            self.__adjust()
        
    
    def __adjust(self):
        self.matrix = np.c_[np.zeros(self.rank), self.matrix]
        self.matrix = np.r_[np.zeros((1, self.rank+1)), self.matrix]
        
        
    def auto_create(self): 
        """
            Auto_create(self): creates random test distance matrix on its own. 
        """
        G= np.zeros((self.rank, self.rank))
        for i in range(self.rank):
            for j in range(self.rank):
                if i==j :
                    continue;
                
                elif(G[i][j]==0):
                    G[i][j]= rd.randint(1, self.rank)
                    G[j][i]= G[i][j]
                else: 
                    continue;
                
        self.matrix = G
        #print(G)
        self.__adjust()
        return self.matrix
    
    def create(self):
        """
            Create(self): users can create their own customized distance matrix. 
        """
        G= np.zeros((self.rank, self.rank))
        for i in range(self.rank):
            for j in range(self.rank):
                if i==j:
                    continue; 
                    
                elif(G[i][j]==0): 
                    G[j][i] = G[i][j] = int(input(f"cost from city{i} to city{j} "))
                else: 
                    continue; 
        
        self.matrix = G
        self.__adjust()
        return self.matrix


#-----------------------------------------------------------------------

class Person(): 
    def __init__(self, person: list):
        self.rank = len(person) 
        self.trait = np.asarray(person)
     
    def __str__(self):
        return f"person.trait =  is {str(self.trait)} "



#----------------------------------------------------------------------------------------------------------------

class Fitness():
    """
     It is to calculate the fitness of route based on the distance matrix. 
    """
    def __init__(self, graph: Graph, route = None):
        """
        Fiteness(self, route= list or rank 0 matrix, graph= adjacency_matrix): creates 
        the object of Fitness class which calculates the routeCost and fitness attached. 
        """
        #self.route= person.trait; 
        self.cost= 0.0; 
        self.fitness=0.0;
        self.costmatrix= graph.matrix;
        self.graph= graph
        #print(self.graph)
        self.route= route
        
        
    def routeCost(self, person: Person = None):
        """
        RouteCost(self): calculates cost related to particular route.
        """
        if person is not None:
            route= person.trait
        else: 
            route = self.route
        
        #print(f"len(route) = {len(route)} , self.graph.rank = {self.graph.rank})
        assert len(route) == self.graph.rank, "rank mismatch"

        
        pathdistance= 0; 
        #print(f"route = {route}, routelen= {len(route)}")
        routelen = len(route)
        for i in range(routelen):
            fromCity= route[i]
            toCity= None
            if i+1<len(route):
                toCity= route[i+1]
            else:
                toCity= route[0]

            pathdistance+= self.costmatrix[fromCity][toCity]

        self.cost= pathdistance
        
        return self.cost
    
#--------------------------------------------------------------------------------------------
class Operators: 
    graph = np.zeros((1, 1)); 
    __totalDist= 0; 
    def __init__(self, graph= None):
        
        """
        Operators(self, graph): It takes graph or distance matrix as its first argument and perform\
        perform various Operations over the routes required for TSP. 
        """
        if graph is not None:
            self.graph= graph
        
    
    def __totaldistance(self, route):
        td= 0.0
        costmatrix= self.graph.matrix
        routelen= len(route)
        assert routlen == self.graph.rank, "rank mismatch"
        
        
        for i in range(routelen-1):
            td+= costmatrix[route[i]][route[i+1]]
            
        td+= costmatrix[route[-1]][route[0]]
        
        return td
            
    
    def crossOver(self, route1, route2, C: float):
        """
            CrossOver(self, route1, route2, c1, r1): 
                it takes two routes as its first two argument and c1 , r1 as parameter to decide crossover length.
        """
        clen = int((C*len(route1) * np.random.random()))
        res = np.zeros((len(route1)), dtype = np.int)
        k= rd.randint(0, len(route1)-1)
        #k=3
        i=k
        j=0
        for _ in range(clen):
            if i == (len(route2)-1):
                i=0
            res[j] = route2[i]
            j+=1
            i+=1
        l=0    
        while j<len(route2):
            if route1[l] not in res:
                res[j] = route1[l]
                j+=1
            
            l+=1
            
        return res    
    
    
    def __injectionPoint(self, route):
        """
            InjectionPoint(self, route): it takes route as its first and only argument. 
                Returns index position as where the injection needs to be done on the basis of distance between two cities. 
                
        """
        dmax= g[route[0]][route[1]]/self.__totalDist
        injPnt= 0
        for i in range(1, len(route)-1):
            d= g[route[i]][route[i+1]]/self.__totalDist
            if d>dmax:
                injPnt = i
                
        return i
    
    def inject(self, route, hfactor= 0.3):
        """
        Inject(self, route, hfactor = 0.3): 
            it takes route as its first argument and hfactor which has default value of 0.3. 
            It injects set of city called vc into the current particle bringing the change in the particle.
        """
        #print(f"route= {route}")
        hfaclen= int(hfactor*len(route));
        #print(f"hfaclen= {hfaclen}"); 
        perm= np.random.permutation(route);
        #print(f"perm= {perm}")
        vc= perm[:hfaclen]
        #print(f"vc= {vc}"); 
        
        inPoint= self.__injectionPoint(route); 
        newroute= np.zeros((route.shape), dtype = np.int); 
        
        #print(f"inPoint= {inPoint}")
        
        #newroute[inPoint:inPoint+hfaclen]= vc
        i= inPoint

        for j in range(hfaclen):
            if i== len(route):
                i=0; 
            newroute[i]= vc[j]
            i+=1
                
                
        #print(f"newroute: {newroute}")
        routeLen= len(route); 
        j=i=0; 
        while i< routeLen:
            if route[i] not in newroute:
                while newroute[j] != 0 and j<routeLen:
                    j+=1
                newroute[j]= route[i]
                i+=1
            else: 
                i+=1; 
            
        print(f"newroute processed under inject: {newroute}")        
        return newroute
        
    def reverse(self, route, s, e): 
        """
         Reverse(self, route, s, e): It does sectional reversal over the route as 1st argument, 
         start point as second and end point as last and third argument. 
        """
        newroute= route
        if s==1: 
            newroute[s-1:e] = route[e-1::-1]
        else:
            newroute[s-1:e] = route[e-1:s-2:-1]
        print(f"newroute processed under reverse: {newroute}")
        return newroute


#---------------------------------------------------------------------------------
class SGO():
    def __init__(self, graph: Graph, person_count: int, persons: list,  generations: int, C: float = 0.4, target: int= None):
        self.graph = graph
        self.person_count = person_count
        self.generations = generations
        self.social_group = [Person(persons[i]) for i in range(person_count)]
        self.sgbest_trait = []; 
        self.sgbest_value = float('inf')
        self.fitness= Fitness(self.graph)
        self.Opr= Operators(self.graph)
        self.C = C
        self.target = target
    
    def printgroup(self):
        for person in self.social_group:
            print(f"{person}, fitness= {self.fitness.routeCost(person)}")
            
    def setgbest(self):
        for person in self.social_group:
            person_fitness = self.fitness.routeCost(person)
            if person_fitness < self.sgbest_value: 
                self.sgbest_value = person_fitness
                self.sgbest_trait = person.trait
                
    def improve_group(self):
        c= self.C
        for i in range(len(self.social_group)):
            person= self.social_group[i]
            old_fitness = self.fitness.routeCost(person)
            r2= self.sgbest_trait;
            r1= person.trait;
            res = self.Opr.crossOver(route1 = r1, route2 = r2, C=c)
            new_fitness = Fitness(self.graph, res).routeCost()
            if new_fitness < old_fitness: 
                self.social_group[i].trait = res
    
    def acquire(self):
        for i in range(len(self.social_group)):
            res= None
            i_r= rd.randint(1, len(self.social_group) -1)
            pi = self.social_group[i]
            
            while i == i_r:
                i_r = rd.randint(1, len(self.social_group) -1) 
                
            Xr= self.social_group[i_r]
            
            fitness_Xr= self.fitness.routeCost(Xr)
            fitness_pi= self.fitness.routeCost(pi)
            
            if fitness_pi < fitness_Xr: 
                res= self.Opr.crossOver(route1= Xr.trait, route2= pi.trait, C= rd.random())
            else: 
                res= self.Opr.crossOver(route1= pi.trait, route2= Xr.trait, C= rd.random())
            
            res= self.Opr.crossOver(route1=res, route2=self.sgbest_trait, C= rd.random())
            
            new_fitness = Fitness(self.graph, res).routeCost()
            
            if new_fitness < fitness_pi:
                self.social_group[i].trait = res
    
    def solve(self):
        iterations = []
        costs = []
        for i in range(self.generations):
            self.setgbest()
            self.improve_group()
            self.setgbest()
            self.acquire()
            #self.printgroup()
            self.setgbest()
            print(f"<<<===============================================================>>>")
            print(f"generation= {i}, gbestroute= {self.sgbest_trait}, gbest_value= {self.sgbest_value}")

            if self.target != None: 
                if self.sgbest_value <= self.target: 
                    break
            costs.append(self.sgbest_value)
            iterations.append(i); 
        
        self.setgbest()
        
        return self.sgbest_trait, self.sgbest_value, iterations, costs
                
                
        
                
                
#----------------------------------------------------------------------------

def createCostMatrix( mat, rank :int):
    arr= mat.split(); 
    arr= np.asarray((arr), dtype= np.int)
    arr= np.reshape(arr, (rank, rank))
    return arr













    
