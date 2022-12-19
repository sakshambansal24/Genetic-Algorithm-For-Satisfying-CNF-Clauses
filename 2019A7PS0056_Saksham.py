from CNF_Creator import *
import numpy as np
import time
import math


# A model is represented by a Numpy array with values 0 or 1, indicating F and T respectively.
def generate_random_model(model_size):
    return np.random.randint(2, size=model_size)

#returns fitness of a model
def calculate_model_fitness(sentence,model):
    valid_clauses = 0
    total_clauses = len(sentence)
    for clause in sentence:
        for symbol in clause:
            if(symbol>0  and model[symbol-1]):
                valid_clauses += 1
                break
            if(symbol<0 and not model[-symbol-1]):
                valid_clauses += 1
                break
    
    return (valid_clauses*100)/total_clauses

#models are sorted in decreasing order based on their fitness value
def fitness(sentence,population):
    pop_fitness = [ {"model":model, "fitness":calculate_model_fitness(sentence,model)} for model in population ]
    sorted_pop_fitness = sorted(pop_fitness, key = lambda x: x["fitness"], reverse=True)
    return sorted_pop_fitness

#from the reverse sorted list a fixed number of parents are selected and returned
def selection(sorted_pop_fitness, parent_nos):
    x = []
    for i in sorted_pop_fitness:
        x.append(i)
        if len(x) == parent_nos:
            break
    return x

# Two point mating
def mate(p1,p2):
    i1 = np.random.randint(0,len(p1))
    i2 = np.random.randint(0,len(p1))
    if i1>i2 : i1,i2 = i2,i1
    child_1 = np.append(p1[:i1], np.append(p2[i1:i2], p1[i2:]))
    child_2 = np.append(p2[:i1], np.append(p1[i1:i2], p2[i2:]))
    return [child_1,child_2]

#mating parents are randomly selected based on probability which has weightage of their fitness value
def select_mating_parents(parent_fitness):
    ratio = [x['fitness'] for x in parent_fitness]
    return random.choices(population= parent_fitness, weights= ratio, k= 2)

#creating offspring from selected mating parents
def crossover(parents_fitness, sorted_pop_fitness):    
    n_child = (len(sorted_pop_fitness) - len(parents_fitness)) // 2
    next_gen = [item['model'] for item in parents_fitness]

    for _ in range(n_child):
        parent1, parent2 = select_mating_parents(sorted_pop_fitness)
        children = mate(parent1['model'], parent2['model'])
        next_gen += children

    return next_gen

#mutation based on a probability which decreases as the number of cycles increase
def mutation(offspring, mutation_prob):
    i = np.random.randint(0,len(offspring))
    if np.random.rand() < mutation_prob:
        offspring[i] = 1-offspring[i]

    return offspring


def print_values(sentence,best_model,best_model_fitness,duration):
    print("\n\n")
    print("Roll No : 2019A7PS0142G")
    print("Number of clauses in CSV file : ",len(sentence))
    print("Best model : ", best_model)
    print("Fitness value of best model : ", best_model_fitness)
    print("Time taken : ", duration , "seconds")
    print('\n\n')

def evolve(sentence, model_size, pop_nos, parent_nos, max_fitness, max_time, mutation_prob,freq):
    # Initializations
    #next_generation is a list of randomly generated models of the size pop_nos
    next_generation = [ generate_random_model(model_size) for i in range(pop_nos)]
    generation_number = -1
    start_time = time.time()
    duration = 0
    fitness_values = 0;
    counter = 0;
    
    # Evolution Loop
    while(duration < max_time):
        generation_number += 1

        population = next_generation
        #storing sorted population based on fitness value
        sorted_pop_fitness = fitness(sentence, population)
        #if max fitness value is achieved function returns
        if(sorted_pop_fitness[0]["fitness"] >= max_fitness):
            break
        #checking for the iteration of algo with same fitness value
        if(sorted_pop_fitness[0]["fitness"]>fitness_values):
            fitness_values = sorted_pop_fitness[0]["fitness"]
            counter = 0
        elif(sorted_pop_fitness[0]["fitness"]==fitness_values):
            counter+=1
            if(counter>=freq):
                 break
        
        #selecting top parent models for returning along with children offspring to next generation
        parents_fitness = selection(sorted_pop_fitness, parent_nos)
        #creating children models through crossover
        non_mutated_next_generation = crossover(parents_fitness, sorted_pop_fitness)
        if(len(sentence)>140):
            mutation_prob -= ((mutation_prob-0.1)*((1-math.exp(-1*counter/60000))/(1+math.exp(-1*counter/60000))))
            # print(mutation_prob,"\n")
        #mutation
        next_generation = mutation(non_mutated_next_generation, mutation_prob)
        duration = time.time() - start_time
    
    if(duration >= max_time):
        exit_result = "Passed Max Time"
    
    best_model = sorted_pop_fitness[0]["model"]
    best_model_fitness = sorted_pop_fitness[0]["fitness"]
    count = 0
    for i in range(len(best_model)):
        count+=1
        if(best_model[i]==1):
            best_model[i]=count
        else:
            best_model[i]=-1*count

    print_values(sentence,best_model,best_model_fitness,duration)
    # return [best_model_fitness,duration]


def main():
    cnfC = CNF_Creator(n=50)
    
    sentence = cnfC.ReadCNFfromCSVfile()
            # print('Sentence from CSV file : ',sentence)
            # GENETIC ALGORITHM
    model_size = 50
    pop_nos = 6
    parent_nos = 2
    max_fitness = 100.00
    max_time = 45
    mutation_prob= 0.5
    freq = 15000

    # sum = 0
    # time = 0
    # for i in range(0,15):
    evolution_results = evolve(sentence, model_size, pop_nos, parent_nos, max_fitness, max_time,mutation_prob,freq)
        # sum+= evolution_results[0]
        # time+=evolution_results[1]

    # print("\n\n")
    # print(sum/15)
    # print(time/15)


    # print('\n\n')
    # print('Roll No : 2020H1030999G')
    # print('Number of clauses in CSV file : ',len(sentence))
    # print('Best model : ',[1, -2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16, -17, 18, 19, -20, 21, -22, 23, -24, 25, 26, -27, -28, 29, -30, -31, 32, 33, 34, -35, 36, -37, 38, -39, -40, 41, 42, 43, -44, -45, -46, -47, -48, -49, -50])
    # print('Fitness value of best model : 99%')
    # print('Time taken : 5.23 seconds')
    # print('\n\n')
    
if __name__=='__main__':
    main()
