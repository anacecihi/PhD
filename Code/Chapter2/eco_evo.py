## Import relevant packages and modules ##
import pandas as pd, math, statistics, random, numpy.random as nura, numpy as np, array as arr, matplotlib.pyplot as plt, matplotlib.patches as mpatches, sys, getopt, time

## Import parameter values from parameters file ##
from parameters import max_time, ancestral_prey, ancestral_parasite, ancestral_predator, infected_prey, infected_predator, mx, my, mz, gx, dx, rx, pop_limit, n_z, dz, S, fy, ky, re, rp, dy, sigma_value_prey, sigma_value_predator, n_loci

## Function for calculating euclidean distance
def compare_genotypes(genotypes_first,genotypes_second):
    total_ones = genotypes_first - genotypes_second
    values = sum(total_ones)
    return values

## Function for appending new genotypes (simulations)
def loop_to_compare_array(the_array, new_genotype):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array_prey
        if(np.array_equal(new_genotype, the_array[jj, 2])):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one (dead or alive) sum one individual
            return the_array
            break
        if jj == the_array.shape[0] - 1:
            the_array = np.append(the_array,[[jj+1, 1, new_genotype]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
            return the_array

## Function for appending new genotypes (storage)
def loop_to_store_array(the_array, new_genotype, initial_time):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array_prey
        if(np.array_equal(new_genotype, the_array[jj, 2]) and (the_array[jj, 1] != 0)):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one (dead or alive) sum one individual
            return the_array
            break
        if jj == the_array.shape[0] - 1:
            the_array = np.append(the_array,[[jj+1, 1, new_genotype, 0, initial_time, 0, 0]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
            return the_array

## Function to remove individuals (storage)
def loop_to_remove(the_array, new_genotype):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array_prey
        if(np.array_equal(new_genotype, the_array[jj, 2]) and (the_array[jj, 1] != 0)):
            the_array[jj, 1] -= 1 # if the new genotype is the same as an existing one (dead or alive) sum one individual
            return the_array
            break

## Function for multiple mutations in loci
def my_mutation(x,mu): # give random number (between 0 and 1) and mutation rate
    return x < mu # condition for using in the function below (the mutation will only happen if the random number is lower that mutation rate) 

def my_mutation_loci(n_loci, mu, initial_array): # give number of loci in genotype, mutation rate, and genotype that is reproducing and may mutate
    mutation_loci = np.zeros(n_loci) # all the posible positions (loci) in which the genotype can mutate
    sum_mutation = 0
    for ii in range(0, n_loci): # check in all loci (positions in genotype)
        mutation_loci[ii] = random.uniform(0,1) # throw a random number (between 0 and 1) for all the positions in genotype
        sum_mutation = sum(1 for x in mutation_loci if my_mutation(x,mu)) # sum all the positions of the genotype were the random number was lower than the mutation rate
    temporal_array = np.array(initial_array) # asign a temporal name to the genotype that may (or may not) mutate
    if(sum_mutation != 0): # if any of the positions had a random number lower that the mutation rate, then mutation will happen
        mut = np.random.choice(n_loci, sum_mutation, replace = False) # pick randomly those loci that will mutate, where sum_mutation is the number of loci mutating, replace = False to avoid using the same locus twice.
        for ii in range(0, sum_mutation): # for all the loci mutating check whether it is a 0 or a 1
            if(temporal_array[mut[ii]] == 0): # if it is a 0, change it to 1
                temporal_array[mut[ii]] = 1
            else: # if it is a 1, change it to 0
                temporal_array[mut[ii]] = 0
    return temporal_array

## Function to calculate the infection values of possible prey/parasite combinations (interactions)
def infection_of_prey(genotype_prey, genotype_parasite):
    part1 = genotype_prey[genotype_parasite == 0] # interaction of current prey and parasite genotypes
    part2 = part1[part1 == 1] # which loci of the prey genotype are resistant to the parasite genotype (1 in the prey matching a 0 in the parasite)
    r = part2.sum() # some all those 1 in prey matching 0 in parasite (calculate partial resistance)
    Qx = float(sigma_value_prey)**float(r) # calculate infection rate according to partial resistance
    return Qx

## Function to calculate the infection values of possible predator/parasite combinations (interactions)
def infection_of_predator(genotype_predator, genotype_parasite):
    part1 = genotype_predator[genotype_parasite == 0] # interaction of current prey and parasite genotypes
    part2 = part1[part1 == 1] # which loci of the predator genotype are resistant to the parasite genotype (1 in the predator matching a 0 in the parasite)
    r = part2.sum()  # some all those 1 in predator matching 0 in parasite (calculate partial resistance)
    Qy = float(sigma_value_predator)**float(r) # calculate infection rate according to partial resistance
    return Qy

## Function to append new infected prey
def loop_infection(array_infected, host_genotype, parasite_genotype):
    for ii in range(0, array_infected.shape[0]): # Go through all the rows in array of infected prey
        if(np.array_equal(array_infected[ii, 2], host_genotype) and np.array_equal(array_infected[ii, 3], parasite_genotype)):
            array_infected[ii, 1] += 1 # if we already have that prey genotype infected by that parasite genotype then just add one individual to that row
            return array_infected
            break
        if ii == array_infected.shape[0] - 1: # else add the new combination of interaction by adding a row with that prey and parasite genotype (one individual)
            array_infected = np.append(array_infected,[[ii+1, 1, host_genotype, parasite_genotype]], axis=0)
            return array_infected

# Create output files
# sys.argv[1] is to specify an argument for reading the first position in the command line.
output_file = open("output_file_" + sys.argv[1] + ".csv", "w")
output_file.write("mx,my,mz,rp,rx,av_effective_prey_genotype,av_effective_predator_genotype,av_effective_parasite_genotype,total_effective_prey_genotype,total_effective_predator_genotype,total_effective_parasite_genotype,total_prey_genotype,total_predator_genotype,total_parasite_genotype,total_all_genotype,av_euclidean_prey,av_euclidean_parasite,av_euclidean_predator,total_euclidean_prey,total_euclidean_parasite,total_euclidean_predator,uninfected_prey,infected_prey,total_prey,uninfected_predator,infected_predator,total_predator,free_parasite,total_parasite,total_abundance,emergence_prey,emergence_status_prey,emergence_predator,emergence_status_predator,emergence_parasite,emergence_status_parasite,extinction_prey,time_prey,extinction_predator,time_predator,extinction_parasite,time_parasite,coexistence,coexistence_pred_prey,coexistence_prey,extinction\n")

time_points = open("time_points_" + sys.argv[1] + ".csv", "w")
time_points.write("time,uninfected_prey,free_parasite,uninfected_predator,infected_prey,infected_predator,total_prey,total_parasite,total_predator,av_effective_prey_genotype,av_effective_parasite_genotype,av_effective_predator_genotype,total_effective_prey_genotype,total_effective_parasite_genotype,total_effective_predator_genotype,total_prey_genotype,total_parasite_genotype,total_predator_genotype,av_euclidean_prey,av_euclidean_parasite,av_euclidean_predator\n")

# Infection parameters
index_to_add = 0 # index of the predator that got infected to keep track of genotype in array of infected individuals

# Genotypes length
initial_prey = np.zeros(n_loci) # initial prey genotype (all loci are initially susceptible, i.e. zeros)
initial_parasite = np.zeros(n_loci) # initial parasite genotype (all loci are initially non-infective, i.e. zeros)
initial_predator = np.zeros(n_loci) # initial predator genotype (all loci are initially susceptible, i.e. zeros)

# Population arrays
np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning) # to avoid warning messages that result due to python version conflict
# Prey
prey = np.array([[0,ancestral_prey,initial_prey]], dtype = object) # array preys for all genotypes (simulation purposes)
store_prey = np.array([[0,ancestral_prey,initial_prey,1,0,0,0]], dtype = object) # array preys for all genotypes (storage purposes)
# Parasite
parasite = np.array([[0,ancestral_parasite,initial_parasite]], dtype = object) # array parasite for all genotypes (simulation purposes)
store_parasite = np.array([[0,ancestral_parasite,initial_parasite,1,0,0,0]], dtype = object) # array parasite for all genotypes (storage purposes)
# Predator
predator = np.array([[0,ancestral_predator,initial_predator]], dtype = object) # array predator for all genotypes (simulation purposes)
store_predator = np.array([[0,ancestral_predator,initial_predator,1,0,0,0]], dtype = object) # array predator for all genotypes (storage purposes)
# Infected prey
infected = np.array([[0,infected_prey,initial_prey,initial_parasite]], dtype = object) # array infected prey (genotypes)
# Infected predator
infected_pred = np.array([[0,infected_predator,initial_predator,initial_parasite]], dtype = object) # array infected predator (genotypes)
mut = np.zeros([n_loci]) # helping array for randomly choosing a location in the genotype that mutates

# Variables for recording time of emergence of all-resistant hosts and all-infective parasites
emergence_st_prey = False
emergence_st_predator = False
emergence_st_parasite = False
emergence_time_prey = max_time
emergence_time_predator = max_time
emergence_time_parasite = max_time

# Variables for recording time of extinctions
extinction_prey = False
extinction_predator = False
extinction_parasite = False
extinction_time_prey = max_time
extinction_time_predator = max_time
extinction_time_parasite = max_time

coexistence = "0"
coexistence_predator_and_prey = "0"
coexistence_prey = "0"
extinction = "0"

# Variables for recording relevant genotypes
total_effective_prey_genotype = 1
total_effective_parasite_genotype = 1
total_effective_predator_genotype = 1

# Data iterations
store_sum_prey = []
store_sum_parasite = []
store_sum_predator = []
store_sum_mutants_prey = []
store_sum_mutants_parasite = []
store_sum_mutants_predator = []
store_sum_infected_prey = []
store_sum_infected_predator = []
store_effective_prey_genotype = []
store_effective_parasite_genotype = []
store_effective_predator_genotype = []
store_current_euclidean_prey = []
store_current_euclidean_parasite = []
store_current_euclidean_predator = []
partial_time = [] # for timing average seconds per iteration

# Continuous time
Time = 0 # total Gillespie time
dt_next_event = 0 # random time step after event occurs (following the Gillespie algorithm). This quantity is summed to the total time (continuos time simulation)
n = 0 # number of steps for recording time points across simulations

while Time < max_time: # SIMULATION STARTS: repeat simulation until reaching max time

    # Optimize arrays (remove extinct genotypes for speeding simulations)
    if(prey.shape[0] != 1):
        prey = prey[prey[:,1] != 0]
    if(infected.shape[0] != 1):
        infected = infected[infected[:,1] != 0]
    if(parasite.shape[0] != 1):
        parasite = parasite[parasite[:,1] != 0]
    if(predator.shape[0] != 1):
        predator = predator[predator[:,1] != 0]
    if(infected_pred.shape[0] != 1):
        infected_pred = infected_pred[infected_pred[:,1] != 0]
    
    # Optimize arrays (show only those that are still alive or were effective)
    if(store_prey.shape[0] != 1):
        store_prey = store_prey[np.logical_not(np.logical_and(store_prey[:,1] == 0, store_prey[:,6] == 0))]
    if(store_parasite.shape[0] != 1):
        store_parasite = store_parasite[np.logical_not(np.logical_and(store_parasite[:,1] == 0, store_parasite[:,6] == 0))]
    if(store_predator.shape[0] != 1):
        store_predator = store_predator[np.logical_not(np.logical_and(store_predator[:,1] == 0, store_predator[:,6] == 0))]
        
###### events uninfected prey ######
    prey_growth = prey[:,1] * gx # prey reproduction
    prey_death = prey[:,1] * dx # prey intrinsic death
    prey_competition = prey[:,1] * (sum(prey[:,1]) + sum(infected[:,1])) * (1 /pop_limit) # prey death due to competition

###### events infected prey ######
    infected_prey_growth = infected[:,1] * gx * rx # infected prey reproduction
    infected_prey_death = infected[:,1] * dx # infected prey intrinsic death
    infected_prey_competition = infected[:,1] * (sum(infected[:,1]) + sum(prey[:,1])) * (1 /pop_limit) # infected prey death due to competition

###### events free-living parasite ######
    infection_prey = np.zeros([parasite.shape[0],prey.shape[0]], dtype = float) # storage array for event
    non_infection_prey = np.zeros([parasite.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,parasite.shape[0]):
        for j in range(0,prey.shape[0]):
            infection_prey[i,j] = parasite[i,1] * prey[j,1] * infection_of_prey(prey[j,2],parasite[i,2]) * S # parasite infects prey
            non_infection_prey[i,j] = parasite[i,1] * prey[j,1] * (1-infection_of_prey(prey[j,2],parasite[i,2])) * S # parasite fails infecting prey
    parasite_death = parasite[:,1] * dz # parasite intrinsic death

###### events uninfected predator ######
    predator_growth = np.zeros([predator.shape[0],prey.shape[0]], dtype = float) # storage array for event
    predator_non_growth = np.zeros([predator.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,predator.shape[0]):
        for j in range(0,prey.shape[0]):
            predator_growth[i,j] = predator[i,1] * prey[j,1] * fy * ky # predator reproduces after feeding
            predator_non_growth[i,j] = predator[i,1] * prey[j,1] * fy * (1-ky) # predator does not reproduce after feeding

    predator_exposure_growth = np.zeros([predator.shape[0],infected.shape[0]], dtype = float) # storage array for event
    predator_exposure_non_growth = np.zeros([predator.shape[0],infected.shape[0]], dtype = float) # storage array for event
    predator_infection_growth = np.zeros([predator.shape[0],infected.shape[0]], dtype = float) # storage array for event
    predator_infection_non_growth = np.zeros([predator.shape[0],infected.shape[0]], dtype = float) # storage array for event
    for i in range(0,predator.shape[0]):
        for j in range(0,infected.shape[0]):
            predator_exposure_growth[i,j] = predator[i,1] * infected[j,1] * fy * (1-infection_of_predator(predator[i,2],infected[j,3])) * re * ky # predator exposed to parasite reproduces
            predator_exposure_non_growth[i,j] = predator[i,1] * infected[j,1] * fy * (1-infection_of_predator(predator[i,2],infected[j,3])) * (1 - (re * ky)) # predator exposed to parasite does not reproduce
            predator_infection_growth[i,j] = predator[i,1] * infected[j,1] * fy * infection_of_predator(predator[i,2],infected[j,3]) * rp * ky # predator infected by parasite reproduces
            predator_infection_non_growth[i,j] = predator[i,1] * infected[j,1] * fy * infection_of_predator(predator[i,2],infected[j,3]) * (1 - (rp * ky)) # predator infected by parasite does not reproduce
    predator_death = predator[:,1] * dy # predator intrinsic death
    
    # events infected predator
    infected_predator_growth = np.zeros([infected_pred.shape[0],prey.shape[0]], dtype = float) # storage array for event
    infected_predator_non_growth = np.zeros([infected_pred.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_pred.shape[0]):
        for j in range(0,prey.shape[0]):
            infected_predator_growth[i,j] = infected_pred[i,1] * prey[j,1] * fy * rp * ky # infected predator reproduces after feeding
            infected_predator_non_growth[i,j] = infected_pred[i,1] * prey[j,1] * fy * (1 - (rp * ky)) # infected predator does not reproduce after feeding
    
    infected_predator_exposure_growth = np.zeros([infected_pred.shape[0],infected.shape[0]], dtype = float) # storage array for event
    infected_predator_exposure_non_growth = np.zeros([infected_pred.shape[0],infected.shape[0]], dtype = float) # storage array for event
    infected_predator_infection_growth = np.zeros([infected_pred.shape[0],infected.shape[0]], dtype = float) # storage array for event
    infected_predator_infection_non_growth = np.zeros([infected_pred.shape[0],infected.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_pred.shape[0]):
        for j in range(0,infected.shape[0]):
            infected_predator_exposure_growth[i,j] = infected_pred[i,1] * infected[j,1] * fy * (1-infection_of_predator(infected_pred[i,2],infected[j,3])) * re * rp * ky # infected predator exposed to the parasite reproduces
            infected_predator_exposure_non_growth[i,j] = infected_pred[i,1] * infected[j,1] * fy * (1-infection_of_predator(infected_pred[i,2],infected[j,3])) * (1 - (re * rp * ky))  # infected predator exposed to the parasite does not reproduce
            infected_predator_infection_growth[i,j] = infected_pred[i,1] * infected[j,1] * fy * infection_of_predator(infected_pred[i,2],infected[j,3]) * rp * rp * ky # infected predator infected by parasite reproduces
            infected_predator_infection_non_growth[i,j] = infected_pred[i,1] * infected[j,1] * fy * infection_of_predator(infected_pred[i,2],infected[j,3]) * (1 - (rp * rp * ky)) # infected predator infected by parasite does not reproduce
    infected_predator_death = infected_pred[:,1] * dy # infected predator intrinsic death
    
    # Sum all events
    sum_events = (prey_growth.sum() + prey_death.sum() + prey_competition.sum() + 
    infected_prey_growth.sum() + infected_prey_death.sum() + infected_prey_competition.sum() + 
    infection_prey.sum() + non_infection_prey.sum() + parasite_death.sum() + 
    predator_growth.sum() + predator_non_growth.sum() + 
    predator_exposure_growth.sum() + predator_exposure_non_growth.sum() + 
    predator_infection_growth.sum() + predator_infection_non_growth.sum() +
    predator_death.sum() + 
    infected_predator_growth.sum() + infected_predator_non_growth.sum() + 
    infected_predator_exposure_growth.sum() + infected_predator_exposure_non_growth.sum()  + 
    infected_predator_infection_growth.sum() + infected_predator_infection_non_growth.sum() + 
    infected_predator_death.sum())

    ## STEP 2 ##
    ## CALCULATE NEXT TIME STEP AND NEXT EVENT ##
    ## GENERATE RANDOM NUMBERS ##
    
    # Next time step
    dt_next_event = np.random.exponential(scale=1/sum_events)

    # Next event
    URN = random.uniform(0,1) # unit-interval uniform random number generator for next event
    P = 0 # for doing cummulative sum in picking the next event

    ## STEP 3 ##
    ## EVENT HAPPENS, UPDATE POPULATION SIZES AND ADD TIME STEP TO TOTAL TIME ##
    
    ############### Uninfected prey ####################
    occurrence = False
    while not occurrence:
        for i in range(0,prey.shape[0]):
            if URN > P and URN <= P + prey_growth[i]/sum_events:
                bx = 1 # uninfected prey increases by one
                bz = 2 # nothing happens to free-living parasite
                by = 2 # nothing happens to predator
                mbx = i # row number in prey array
                occurrence = True
                break
            P += prey_growth[i]/sum_events #! use += to modify in place
            
            if URN > P and URN <= P + prey_death[i]/sum_events:
                bx = 0 # uninfected prey decreases by one
                bz = 2 # nothing happens to free-living parasite
                by = 2 # nothing happens to predator
                mbx = i
                occurrence = True
                break
            P += prey_death[i]/sum_events
        
            if URN > P and URN <= P + prey_competition[i]/sum_events:
                bx = 0 # uninfected prey decreases by one
                bz = 2 # nothing happens to free-living parasite
                by = 2 # nothing happens to predator
                mbx = i
                occurrence = True
                break
            P += prey_competition[i]/sum_events
        
        if occurrence:
            break

        ############### Infected prey #################
        for i in range(0,infected.shape[0]):
            if URN > P and URN <= P + infected_prey_growth[i]/sum_events:
                bx = 6 # infected prey reproduces
                bz = 2 # nothing happens to free-living parasite
                by = 2 # nothing happens to predator
                mbi = i
                occurrence = True
                break
            P += infected_prey_growth[i]/sum_events

            if URN > P and URN <= P + infected_prey_death[i]/sum_events:
                bx = 5 # infected prey decreases by one
                bz = 2 # nothing happens to free-living parasite
                by = 2 # nothing happens to predator
                mbi = i
                occurrence = True
                break
            P += infected_prey_death[i]/sum_events

            if URN > P and URN <= P + infected_prey_competition[i]/sum_events:
                bx = 5 # infected prey decreases by one
                bz = 2 # nothing happens to free-living parasite
                by = 2 # nothing happens to predator
                mbi = i
                occurrence = True
                break
            P += infected_prey_competition[i]/sum_events
        
        if occurrence:
            break
                                            
        ################ Free-living parasite ####################
        for i in range(0,parasite.shape[0]):
            for j in range(0,prey.shape[0]):
                if URN > P and URN <= P + infection_prey[i,j]/sum_events:
                    bx = 3 # prey is carrying a parasite (now it is infected)
                    bz = 0 # parasite gets in the prey
                    by = 2 # nothing happens to predator
                    mbx = j
                    mbz = i
                    occurrence = True
                    break
                P += infection_prey[i,j]/sum_events
            
                if URN > P and URN <= P + non_infection_prey[i,j]/sum_events:
                    bx = 2 # nothing happens to prey (it is not infected)
                    bz = 2 #  nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mbx = j
                    mbz = i
                    occurrence = True
                    break
                P += non_infection_prey[i,j]/sum_events

        if occurrence:
            break

        for i in range(0,parasite.shape[0]):
            if URN > P and URN <= P + parasite_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 0 # free-living parasite decreases by one
                by = 2 # nothing happens to predator
                mbz = i
                occurrence = True
                break
            P += parasite_death[i]/sum_events
        
        if occurrence:
            break

        ################ Uninfected predator #####################
        for i in range(0,predator.shape[0]):
            for j in range(0,prey.shape[0]):
                if URN > P and URN <= P + predator_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 1 # predator increases by one
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += predator_growth[i,j]/sum_events

                if URN > P and URN <= P + predator_non_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += predator_non_growth[i,j]/sum_events

        if occurrence:
            break
        
        for i in range(0,predator.shape[0]):          
            if URN > P and URN <= P + predator_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 2 # nothing happens to free-living parasite
                by = 0 # predator decreases by one  
                mby = i
                occurrence = True
                break
            P += predator_death[i]/sum_events
        
        if occurrence:
            break

        for i in range(0,predator.shape[0]):
            for j in range(0,infected.shape[0]):
                if URN > P and URN <= P + predator_exposure_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 1 # predator increases by one
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += predator_exposure_growth[i,j]/sum_events

                if URN > P and URN <= P + predator_exposure_non_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += predator_exposure_non_growth[i,j]/sum_events

                if URN > P and URN <= P + predator_infection_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
                    by = 3 # predator is infected and increases by one
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += predator_infection_growth[i,j]/sum_events
                            
                if URN > P and URN <= P + predator_infection_non_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
                    by = 4 # predator is infected and does not reproduce
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += predator_infection_non_growth[i,j]/sum_events

        if occurrence:
            break
                            
        ################ Infected predator #####################
        for i in range(0,infected_pred.shape[0]):
            for j in range(0,prey.shape[0]):
                if URN > P and URN <= P + infected_predator_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 6 # predator increases by one
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += infected_predator_growth[i,j]/sum_events
                                
                if URN > P and URN <= P + infected_predator_non_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += infected_predator_non_growth[i,j]/sum_events
        
        if occurrence:
            break

        for i in range(0,infected_pred.shape[0]):
            for j in range(0,infected.shape[0]):
                if URN > P and URN <= P + infected_predator_exposure_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 6 # predator increases by one
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_exposure_growth[i,j]/sum_events
                                    
                if URN > P and URN <= P + infected_predator_exposure_non_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_exposure_non_growth[i,j]/sum_events
                                        
                if URN > P and URN <= P + infected_predator_infection_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
                    by = 6 # predator increases by one
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_infection_growth[i,j]/sum_events
                                        
                if URN > P and URN <= P + infected_predator_infection_non_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
                    by = 2 # nothing happens to predator
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_infection_non_growth[i,j]/sum_events

        if occurrence:
            break

        for i in range(0,infected_pred.shape[0]):
            if URN > P and URN <= P + infected_predator_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 2 # nothing happens to free-living parasite
                by = 5 # infected predator decreases by one  
                mby = i
                occurrence = True
                break
            P += infected_predator_death[i]/sum_events
    
        if occurrence:
            break

##################### PREY EVENTS #########################
    if(bx == 1): # prey reproduces
        new_prey = np.array(my_mutation_loci(n_loci, mx, prey[mbx,2])) # new genotype that results after reproduction (may have a mutation or not)
        prey = loop_to_compare_array(prey, new_prey) # either append new genotype or add individual if already exists
        store_prey = loop_to_store_array(store_prey, new_prey, Time)
        temporal_prey = str(new_prey) # check if offspring was super-resistant
        if(temporal_prey == str(np.ones(n_loci)) and emergence_time_prey == max_time):
            emergence_time_prey = Time
            emergence_st_prey = True

    elif(bx == 0): # prey dies
        prey[mbx,1] -= 1 # decrease one prey individual
        store_prey = loop_to_remove(store_prey, prey[mbx,2])
    
    elif(bx == 3):# parasite infects prey
        prey[mbx,1] -= 1  # prey gets infected
        infected = loop_infection(infected, prey[mbx,2], parasite[mbz,2]) # either append new genotype infected or add individual if already exists

    elif(bx == 5): # infected prey dies
        infected[mbi,1] -= 1 # decrease one infected prey from that particular row (where event happened)
        store_prey = loop_to_remove(store_prey, infected[mbi,2])

    elif(bx == 6): # infected prey reproduces
        new_infected_prey = np.array(my_mutation_loci(n_loci, mx, infected[mbi,2])) # new genotype that results after reproduction (may have a mutation or not)
        prey = loop_to_compare_array(prey, new_infected_prey) # either append new genotype or add individual if already exists (it is added to uninfected prey array beacuse parasite is not transmitted from parent to offspring)
        store_prey = loop_to_store_array(store_prey, new_infected_prey, Time)
        temporal_prey = str(new_infected_prey) # check if offspring was super-resistant
        if(temporal_prey == str(np.ones(n_loci)) and emergence_time_prey == max_time):
            emergence_time_prey = Time
            emergence_st_prey = True

################### PARASITE EVENTS #######################
    if(bz == 0): # free-living parasite dies
        parasite[mbz,1] -= 1
        store_parasite = loop_to_remove(store_parasite, parasite[mbz,2])
    
    elif(bz == 1): # parasite may or may not infect an uninfected predator
        new_parasite = np.zeros(n_z)
        for i in range(0,n_z): # repeat this for all parasite offspring
            new_parasite = np.array(my_mutation_loci(n_loci, mz, infected[mbi,3])) # check for mutations
            parasite = loop_to_compare_array(parasite, new_parasite) # add new genotype (may contain mutation or may not)
            store_parasite = loop_to_store_array(store_parasite, new_parasite, Time)
        # If all-resistant or all-infective genotypes emerged, record the id and emergence time
        temporal_parasite = str(new_parasite) # check if offspring was super-infective
        if(temporal_parasite == str(np.ones(n_loci)) and emergence_time_parasite == max_time):
            emergence_time_parasite = Time
            emergence_st_parasite = True
    
################### PREDATOR EVENTS ########################
    if(by == 1): # predator reproduces
        new_predator = np.array(my_mutation_loci(n_loci, my, predator[mby,2])) # check for mutation in predator offspring
        predator = loop_to_compare_array(predator, new_predator) # add new genotype (may contain mutation or may not)
        store_predator = loop_to_store_array(store_predator, new_predator, Time)
        temporal_predator = str(new_predator) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True

    elif(by == 0): # predator dies
        predator[mby,1] -= 1 # uninfected predator of that genotype (row in uninfected predator array) decreases by one
        store_predator = loop_to_remove(store_predator, predator[mby,2])

    elif(by == 3): # predator gets infected and reproduces
        # Infection part
        predator[mby,1] -= 1
        infected_pred = loop_infection(infected_pred, predator[mby,2], infected[mbi,3]) 
        # Reproduction part
        new_predator = np.array(my_mutation_loci(n_loci, my, predator[mby,2])) # the predator is also reproducing, create new genotype (may or may not have mutation)
        predator = loop_to_compare_array(predator, new_predator) # add new genotype to the uninfected predator array (parasite is not transmitted from parent to offspring)
        store_predator = loop_to_store_array(store_predator, new_predator, Time)
        temporal_predator = str(new_predator) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True

    elif(by == 4): # predator gets infected and does not reproduce
        # Infection part
        predator[mby,1] -= 1
        infected_pred = loop_infection(infected_pred, predator[mby,2], infected[mbi,3])

    elif(by == 5): # infected predator dies
        infected_pred[mby,1] -= 1
        store_predator = loop_to_remove(store_predator, infected_pred[mby,2])
        
    elif(by == 6): # infected predator reproduces
        # Reproduction part
        new_infected_pred = np.array(my_mutation_loci(n_loci, my, infected_pred[mby,2])) # check for mutations in genotype of offspring
        predator = loop_to_compare_array(predator, new_infected_pred) # add genotype to uninfected predator array
        store_predator = loop_to_store_array(store_predator, new_infected_pred, Time)
        temporal_predator = str(new_infected_pred) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True

    # Advance a step in time
    Time += dt_next_event # continuous time simulation

# Record relevant genotypes (same percentages for the three entities)
# If they have a "1", they are effective genotypes
    for i in range(0, store_prey.shape[0]): # goes extinct
        if(store_prey[i,1] == 0 and store_prey[i,3] == 1):
            store_prey[i,3] = 0
            store_prey[i,5] = Time
            store_prey[i,6] = Time - store_prey[i,4]
    
    for i in range(0, store_parasite.shape[0]): # goes extinct
        if(store_parasite[i,1] == 0 and store_parasite[i,3] == 1):
            store_parasite[i,3] = 0
            store_parasite[i,5] = Time
            store_parasite[i,6] = Time - store_parasite[i,4]

    for i in range(0, store_predator.shape[0]): # goes extinct
        if(store_predator[i,1] == 0 and store_predator[i,3] == 1):
            store_predator[i,3] = 0
            store_predator[i,5] = Time
            store_predator[i,6] = Time - store_predator[i,4]

    for i in range(0, store_prey.shape[0]):
        if sum(store_prey[:,1]) >= 1: # becomes effective genotype
            if(store_prey[i,1] >= (sum(store_prey[:,1]) * 0.02) and store_prey[i,3] == 0):
                total_effective_prey_genotype += 1
                store_prey[i,3] = 1

    for i in range(0, store_parasite.shape[0]):
        if sum(store_parasite[:,1]) >= 1: # becomes effective genotype
            if(store_parasite[i,1] >= (sum(store_parasite[:,1]) * 0.02) and store_parasite[i,3] == 0):
                total_effective_parasite_genotype += 1
                store_parasite[i,3] = 1

    for i in range(0, store_predator.shape[0]):
        if sum(store_predator[:,1]) >= 1: # becomes effective genotype
            if(store_predator[i,1] >= (sum(store_predator[:,1]) * 0.02) and store_predator[i,3] == 0):
                total_effective_predator_genotype += 1
                store_predator[i,3] = 1

    effective_prey_genotype = sum(store_prey[:,3]) # number of prey genotypes that are effective
    effective_parasite_genotype = sum(store_parasite[:,3]) # number of parasite genotypes that are effective
    effective_predator_genotype = sum(store_predator[:,3]) # number of predator genotypes that are effective
    store_effective_prey_genotype.append(effective_prey_genotype)
    store_effective_parasite_genotype.append(effective_parasite_genotype)
    store_effective_predator_genotype.append(effective_predator_genotype)

    ###### Euclidean distance prey ######
    current_euclidean_prey = 0
    if sum(store_prey[:,1]) >= 1: # if there are prey genotypes
        temporal_store_prey = store_prey[store_prey[:,3] == 1] # use only the effective genoytpes
        mesh_prey = np.array(np.meshgrid(temporal_store_prey[:,2], temporal_store_prey[:,2]))
        genotypes_prey = mesh_prey.T.reshape(-1, 2) # create matrix: first column is prey genotypes and second column is the same prey genotypes
        values = np.zeros([genotypes_prey.shape[0]])
        for i in range(0,genotypes_prey.shape[0]):
            values[i] = compare_genotypes(genotypes_prey[i,0], genotypes_prey[i,1]) # compare number of ones in all the possible combinations of the prey genotypes
        total_values = []
        for i in range(0,genotypes_prey.shape[0]):
            if values[i] > 0: # account only the values from the "triangle" of the matrix
                total_values.append(values[i])
                current_euclidean_prey = statistics.mean(total_values)
    store_current_euclidean_prey.append(current_euclidean_prey)

    ###### Euclidean distance parasite ######
    current_euclidean_parasite = 0
    if sum(store_parasite[:,1]) >= 1: # if there are parasite genotypes in the free-living stage
        temporal_store_parasite = store_parasite[store_parasite[:,3] == 1] # use only the effective genoytpes
        mesh_parasite = np.array(np.meshgrid(temporal_store_parasite[:,2], temporal_store_parasite[:,2]))
        genotypes_parasite = mesh_parasite.T.reshape(-1, 2) # create matrix: first column is parasite genotypes and second column is the same parasite genotypes
        values = np.zeros([genotypes_parasite.shape[0]])
        for i in range(0,genotypes_parasite.shape[0]):
            values[i] = compare_genotypes(genotypes_parasite[i,0], genotypes_parasite[i,1]) # compare number of ones in all the possible combinations of the parasite genotypes
        total_values = []
        for i in range(0,genotypes_parasite.shape[0]):
            if values[i] > 0: # account only the values from the "triangle" of the matrix
                total_values.append(values[i])
                current_euclidean_parasite = statistics.mean(total_values)
    store_current_euclidean_parasite.append(current_euclidean_parasite)

    ###### Euclidean distance predator ######
    current_euclidean_predator = 0
    if sum(store_predator[:,1]) >= 1: # if there are predator genotypes
        temporal_store_predator = store_predator[store_predator[:,3] == 1] # use only the effective genoytpes
        mesh_predator = np.array(np.meshgrid(temporal_store_predator[:,2], temporal_store_predator[:,2]))
        genotypes_predator = mesh_predator.T.reshape(-1, 2) # create matrix: first column is predator genotypes and second column is the same predator genotypes
        values = np.zeros([genotypes_predator.shape[0]])
        for i in range(0,genotypes_predator.shape[0]):
            values[i] = compare_genotypes(genotypes_predator[i,0], genotypes_predator[i,1]) # compare number of ones in all the possible combinations of the predator genotypes
        total_values = []
        for i in range(0,genotypes_predator.shape[0]):
            if values[i] > 0: # account only the values from the "triangle" of the matrix
                total_values.append(values[i])
                current_euclidean_predator = statistics.mean(total_values)
    store_current_euclidean_predator.append(current_euclidean_predator)

    # Update population sizes
    # free-living individuals
    sum_mutants_prey = np.sum(prey[:,1]) # uninfected prey
    sum_mutants_parasite = np.sum(parasite[:,1]) # free-living parasites
    sum_mutants_predator = np.sum(predator[:,1]) # uninfected predator
    # infected hosts
    sum_infected_prey = np.sum(infected[:,1]) # infected prey
    sum_infected_predator = np.sum(infected_pred[:,1]) # infected predator
    # total
    sum_prey = sum_mutants_prey + sum_infected_prey # all prey (uninfected and infected)
    sum_parasite = sum_mutants_parasite + sum_infected_prey # all parasites (free-living)
    sum_predator = sum_mutants_predator + sum_infected_predator # all predator (uninfected and infected)
    # store free-living individuals
    store_sum_mutants_prey.append(sum_mutants_prey) # uninfected prey
    store_sum_mutants_parasite.append(sum_mutants_parasite) # free-living parasites
    store_sum_mutants_predator.append(sum_mutants_predator) # uninfected predators
    # store infected individuals
    store_sum_infected_prey.append(sum_infected_prey) # infected prey
    store_sum_infected_predator.append(sum_infected_predator) # infected predators
    # store total individuals
    store_sum_prey.append(sum_prey) # all preys
    store_sum_parasite.append(sum_parasite) # free-living parasites
    store_sum_predator.append(sum_predator) # all predators

# if prey, parasite, and predator go extinct, stop simulations (also record which one went extinct and the time)
    if(sum_prey <= 0 and not extinction_prey):
        extinction_prey = True
        extinction_time_prey = Time

    if(sum_parasite <= 0 and not extinction_parasite):
        extinction_parasite = True
        extinction_time_parasite = Time

    if(sum_predator <= 0 and not extinction_predator):
        extinction_predator = True
        extinction_time_predator = Time

    # Record coexistence/extinctions
    if not extinction_prey and not extinction_parasite and not extinction_predator:
        coexistence = "1"
        coexistence_predator_and_prey = "0"
        coexistence_prey = "0"
        extinction = "0"
    elif not extinction_prey and extinction_parasite and not extinction_predator:
        coexistence = "0"
        coexistence_predator_and_prey = "1"
        coexistence_prey = "0"
        extinction = "0"
    elif not extinction_prey and extinction_parasite and extinction_predator:
        coexistence = "0"
        coexistence_predator_and_prey = "0"
        coexistence_prey = "1"
        extinction = "0"
    elif extinction_prey and extinction_parasite and extinction_predator:        
        coexistence = "0"
        coexistence_predator_and_prey = "0"
        coexistence_prey = "0"
        extinction = "1"

    if Time > n:
        time_points.write(str(Time) + "," + str(sum_mutants_prey) + "," + str(sum_mutants_parasite) + "," + str(sum_mutants_predator) + "," + str(sum_infected_prey) + ","  + str(sum_infected_predator) + "," + str(sum_prey) + "," + str(sum_parasite) + "," + str(sum_predator) + "," + str(effective_prey_genotype) + "," + str(effective_parasite_genotype) + "," + str(effective_predator_genotype) + "," + str(total_effective_prey_genotype) + "," + str(total_effective_parasite_genotype) + "," + str(total_effective_predator_genotype) + "," + str(store_prey.shape[0]) + "," + str(store_parasite.shape[0]) + "," + str(store_predator.shape[0]) + "," + str(current_euclidean_prey) + "," + str(current_euclidean_parasite) + "," + str(current_euclidean_predator) + "\n")
        n += 0.1

    if(sum_parasite <= 0 and sum_predator <= 0): # break while loop if only the prey remains
        break
        
# Simulation finishes
###### Euclidean distance prey ######
mesh_prey = np.array(np.meshgrid(store_prey[:,2], store_prey[:,2]))
genotypes_prey = mesh_prey.T.reshape(-1, 2) # create matrix: first column is prey genotypes and second column is the same prey genotypes
values = np.zeros([genotypes_prey.shape[0]])
for i in range(0,genotypes_prey.shape[0]):
    values[i] = compare_genotypes(genotypes_prey[i,0], genotypes_prey[i,1]) # compare number of ones in all the possible combinations of the prey genotypes

combinations_prey = np.zeros([genotypes_prey.shape[0],3],dtype=object)
total_values = []
total_euclidean_prey = 0
for i in range(0,genotypes_prey.shape[0]):
    combinations_prey[i] = np.array([genotypes_prey[i,0],genotypes_prey[i,1],values[i]]) # comparison of genotypes (only those with differences in the "triangle of the matrix")
    if values[i] > 0:
        total_values.append(values[i])
        total_euclidean_prey = statistics.mean(total_values)

###### Euclidean distance parasite ######
mesh_parasite = np.array(np.meshgrid(store_parasite[:,2], store_parasite[:,2]))
genotypes_parasite = mesh_parasite.T.reshape(-1, 2) # create matrix: first column is parasite genotypes and second column is the same parasite genotypes
values = np.zeros([genotypes_parasite.shape[0]])
for i in range(0,genotypes_parasite.shape[0]):
    values[i] = compare_genotypes(genotypes_parasite[i,0], genotypes_parasite[i,1]) # compare number of ones in all the possible combinations of the parasite genotypes

combinations_parasite = np.zeros([genotypes_parasite.shape[0],3],dtype=object)
total_values = []
total_euclidean_parasite = 0
for i in range(0,genotypes_parasite.shape[0]):
    combinations_parasite[i] = np.array([genotypes_parasite[i,0],genotypes_parasite[i,1],values[i]]) # comparison of genotypes (only those with differences in the "triangle of the matrix")
    if values[i] > 0:
        total_values.append(values[i])
        total_euclidean_parasite = statistics.mean(total_values)

###### Euclidean distance predator ######
mesh_predator = np.array(np.meshgrid(store_predator[:,2], store_predator[:,2]))
genotypes_predator = mesh_predator.T.reshape(-1, 2) # create matrix: first column is predator genotypes and second column is the same predator genotypes
values = np.zeros([genotypes_predator.shape[0]])
for i in range(0,genotypes_predator.shape[0]):
    values[i] = compare_genotypes(genotypes_predator[i,0], genotypes_predator[i,1]) # compare number of ones in all the possible combinations of the predator genotypes

combinations_predator = np.zeros([genotypes_predator.shape[0],3],dtype=object)
total_values = []
total_euclidean_predator = 0
for i in range(0,genotypes_predator.shape[0]):
    combinations_predator[i] = np.array([genotypes_predator[i,0],genotypes_predator[i,1],values[i]]) # comparison of genotypes (only those with differences in the "triangle of the matrix")
    if values[i] > 0:
        total_values.append(values[i])
        total_euclidean_predator = statistics.mean(total_values)

# Save free-living individuals
av_prey = sum(store_sum_mutants_prey[:]) / len(store_sum_mutants_prey)
av_parasite = sum(store_sum_mutants_parasite[:]) / len(store_sum_mutants_parasite)
av_predator = sum(store_sum_mutants_predator[:]) / len(store_sum_mutants_predator)
# Save infected individuals
av_infected_prey = sum(store_sum_infected_prey[:]) / len(store_sum_infected_prey)
av_infected_predator = sum(store_sum_infected_predator[:]) / len(store_sum_infected_predator)
# Save total individuals
av_sum_prey = sum(store_sum_prey[:]) / len(store_sum_prey)
av_sum_parasite = sum(store_sum_parasite[:]) / len(store_sum_parasite)
av_sum_predator = sum(store_sum_predator[:]) / len(store_sum_predator)
sum_ind = (int(av_sum_prey) + int(av_sum_predator) + int(av_sum_parasite))
# Save total genotypes
t_types_prey = store_prey.shape[0] # number of rows in prey (number of historical genotypes)
t_types_parasite = store_parasite.shape[0]  # number of rows in free-living parasite (number of historical genotypes)
t_types_predator = store_predator.shape[0]  # number of rows in predator (number of historical genotypes)
total_genotype = t_types_prey + t_types_predator + t_types_parasite
# Save average effective genotypes
av_effective_prey_genotype = sum(store_effective_prey_genotype[:]) / len(store_effective_prey_genotype)
av_effective_parasite_genotype = sum(store_effective_parasite_genotype[:]) / len(store_effective_parasite_genotype)
av_effective_predator_genotype = sum(store_effective_predator_genotype[:]) / len(store_effective_predator_genotype)
# Save average euclidean distance
av_euclidean_prey = sum(store_current_euclidean_prey[:]) / len(store_current_euclidean_prey)
av_euclidean_parasite = sum(store_current_euclidean_parasite[:]) / len(store_current_euclidean_parasite)
av_euclidean_predator = sum(store_current_euclidean_predator[:]) / len(store_current_euclidean_predator)

# Record coexistence/extinctions
if not extinction_prey and not extinction_parasite and not extinction_predator:
    coexistence = "1"
    coexistence_predator_and_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_parasite and not extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "1"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_parasite and extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_prey = "1"
    extinction = "0"
elif extinction_prey and extinction_parasite and extinction_predator:        
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_prey = "0"
    extinction = "1"

# Save effective genotypes lifespans that survived
for i in range(0, store_prey.shape[0]): # goes extinct
    if(store_prey[i,1] != 0 and store_prey[i,3] == 1):
        store_prey[i,5] = Time
        store_prey[i,6] = Time - store_prey[i,4]

for i in range(0, store_parasite.shape[0]): # goes extinct
    if(store_parasite[i,1] != 0 and store_parasite[i,3] == 1):
        store_parasite[i,5] = Time
        store_parasite[i,6] = Time - store_parasite[i,4]

for i in range(0, store_predator.shape[0]): # goes extinct
    if(store_predator[i,1] != 0 and store_predator[i,3] == 1):
        store_predator[i,5] = Time
        store_predator[i,6] = Time - store_predator[i,4]

# Save output
output_file.write(str(mx) + "," + str(my) + "," + str(mz) + "," + str(rp) + "," + str(rx) + "," + str(av_effective_prey_genotype) + "," + str(av_effective_predator_genotype) + "," + str(av_effective_parasite_genotype) + "," + str(total_effective_prey_genotype) + "," + str(total_effective_predator_genotype) + "," + str(total_effective_parasite_genotype) + "," + str(t_types_prey) + "," + str(t_types_predator) + "," + str(t_types_parasite) + "," + str(total_genotype) + "," + str(av_euclidean_prey) + "," + str(av_euclidean_parasite) + "," + str(av_euclidean_predator) + "," + str(total_euclidean_prey) + "," + str(total_euclidean_parasite) + "," + str(total_euclidean_predator) + "," + str(int(av_prey)) + "," + str(int(av_infected_prey)) + "," + str(int(av_sum_prey)) + "," + str(int(av_predator)) + "," + str(int(av_infected_predator)) + "," + str(int(av_sum_predator)) + "," + str(int(av_parasite)) + "," + str(int(av_sum_parasite)) + "," + str(sum_ind) + "," + str(emergence_time_prey) + "," + str(emergence_st_prey) + "," + str(emergence_time_predator) + "," + str(emergence_st_predator) + "," + str(emergence_time_parasite) + "," + str(emergence_st_parasite) + "," + str(extinction_prey) + "," + str(extinction_time_prey) + "," + str(extinction_predator) + "," + str(extinction_time_predator) + "," + str(extinction_parasite) + "," + str(extinction_time_parasite) + "," + str(coexistence) + "," + str(coexistence_predator_and_prey) + "," + str(coexistence_prey) + "," + str(extinction) + "\n")

# Close files
output_file.close()
time_points.close()