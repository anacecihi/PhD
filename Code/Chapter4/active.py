## Import relevant packages and modules ##
import pandas as pd, math, statistics, random, numpy.random as nura, numpy as np, array as arr, matplotlib.pyplot as plt, matplotlib.patches as mpatches, sys, getopt, time
from csv import writer

# initial population sizes
uninfected_prey = 800 # uninfected prey
uninfected_predator = 100 # uninfected predator
active_parasite = 1000 # free-living parasites
trophic_parasite = 1000 # free-living parasites
infected_prey_pop = 0 # at the beginning of the simulation, all prey are uninfected
infected_predator_active_pop = 0 # at the beginning of the simulation, all predator are uninfected
infected_predator_trophic_pop = 0 # at the beginning of the simulation, all predator are uninfected

# prey parameters 
gx = 2.0 # growth rate
dx = 0.1 # intrinsic death
pop_limit = 2000 # population limit

# parasite parameters
n_z_active = 5 # number of offspring per reproduction event
n_z_trophic = 12 # number of offspring per reproduction event
dz = 0.09 # intrinsic death

#predator parameters
fy = 0.01 # predation rate
ky = 0.2 # reproduction rate#
dy = 1.0 # intrinsic death

# infection
E = 0.01 # contact rate between predators
S = 0.0005 # scaling factor predator-parasite
T = 0.0005 # scaling factor prey-parasite
rp1 = float(sys.argv[1])
rp2 = float(sys.argv[2])
rx = float(sys.argv[3])
sigma_value_predator = 0.85
sigma_value_prey = 0.85

# genotypes
n_loci = 10 # total number of loci in genotypes
mx = 0.00002 # mutation rate prey
mz = 0.00006 # mutation rate parasite
my = 0.00005 # mutation rate predator

# recording time
recording_time = 200 # time for recording the abundance of each entity
max_time = 1000 # run algorithm until reaching max time

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

# Genotypes length
initial_prey = np.zeros(n_loci) # initial prey genotype (all loci are initially susceptible, i.e. zeros)
initial_predator = np.zeros(n_loci) # initial predator genotype (all loci are initially susceptible, i.e. zeros)
initial_active = np.zeros(n_loci) # initial parasite genotype (all loci are initially non-infective, i.e. zeros)
initial_trophic = np.zeros(n_loci) # initial parasite genotype (all loci are initially non-infective, i.e. zeros)
mut = np.zeros([n_loci]) # helping array for randomly choosing a location in the genotype that mutates

# Population arrays
np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning) # to avoid warning messages that result due to python version conflict
# Prey
prey = np.array([[0,uninfected_prey,initial_prey]], dtype = object) # array preys for all genotypes (simulation purposes)
store_prey = np.array([[0,uninfected_prey,initial_prey,1,0,0,0]], dtype = object) # array preys for all genotypes (storage purposes)
# Active parasite
active = np.array([[0,active_parasite,initial_active]], dtype = object) # array parasite for all genotypes (simulation purposes)
store_active = np.array([[0,active_parasite,initial_active,1,0,0,0]], dtype = object) # array parasite for all genotypes (storage purposes)
# Trophic parasite
trophic = np.array([[0,trophic_parasite,initial_trophic]], dtype = object) # array parasite for all genotypes (simulation purposes)
store_trophic = np.array([[0,trophic_parasite,initial_trophic,1,0,0,0]], dtype = object) # array parasite for all genotypes (storage purposes)
# Predator
predator = np.array([[0,uninfected_predator,initial_predator]], dtype = object) # array predator for all genotypes (simulation purposes)
store_predator = np.array([[0,uninfected_predator,initial_predator,1,0,0,0]], dtype = object) # array predator for all genotypes (storage purposes)
# Infected prey
infected_prey = np.array([[0,infected_prey_pop,initial_prey,initial_trophic]], dtype = object) # array infected prey (genotypes)
# Infected predator active
infected_predator_active = np.array([[0,infected_predator_active_pop,initial_predator,initial_active]], dtype = object) # array infected predator (genotypes)
# Infected predator trophic
infected_predator_trophic = np.array([[0,infected_predator_trophic_pop,initial_predator,initial_trophic]], dtype = object) # array infected predator (genotypes)

# Variables for recording time of emergence of all-resistant hosts and all-infective parasites
emergence_st_prey = False
emergence_st_predator = False
emergence_st_active = False
emergence_st_trophic = False
emergence_time_prey = max_time
emergence_time_predator = max_time
emergence_time_active = max_time
emergence_time_trophic = max_time

# Variables for recording time of extinctions
extinction_prey = False
extinction_predator = False
extinction_active = False
extinction_trophic = False
extinction_time_prey = max_time
extinction_time_predator = max_time
extinction_time_active = max_time
extinction_time_trophic = max_time

# Record persistance parasites
persistance_neither = "0"
persistance_both = "0"
persistance_active = "0"
persistance_trophic = "0"

# Variables for recording coexistence
coexistence = "0"
coexistence_predator_prey_trophic = "0"
coexistence_predator_prey_active = "0"
coexistence_predator_prey = "0"
coexistence_prey = "0"
extinction = "0"

# Abundance genotypes and lifespans
store_sum_prey = []
store_sum_predator = []
store_sum_parasite = []
store_sum_active_abundance = []
store_sum_trophic_abundance = []
store_active_rel_abundance = []
store_trophic_rel_abundance = []
store_effective_prey = []
store_effective_predator = []
store_effective_active = []
store_effective_trophic = []
lifespan_prey = []
lifespan_predator = []
lifespan_active = []
lifespan_trophic = []

# Abundance genotypes and lifespans
active_rel_abundance = 0
trophic_rel_abundance = 0
av_sum_prey = 0
av_sum_predator = 0
av_sum_parasite = 0
av_active_abundance = 0
av_trophic_abundance = 0
av_active_rel_abundance = 0
av_trophic_rel_abundance = 0
av_effective_prey = 0
av_effective_predator = 0
av_effective_active = 0
av_effective_trophic = 0
av_lifespan_prey = 0
av_lifespan_predator = 0
av_lifespan_active = 0
av_lifespan_trophic = 0

# Continuous time
Time = 0 # initial Gillespie time
dt_next_event = 0 # random time step after event occurs (following the Gillespie algorithm). This quantity is summed to the total time (continuos time simulation)
n = 0 # number of steps for recording time points across simulations

while Time < max_time: # SIMULATION STARTS: repeat simulation until reaching max time

    # Optimize arrays (remove extinct genotypes for speeding simulations)
    if(prey.shape[0] != 1):
        prey = prey[prey[:,1] != 0]
    if(predator.shape[0] != 1):
        predator = predator[predator[:,1] != 0]
    if(active.shape[0] != 1):
        active = active[active[:,1] != 0]
    if(trophic.shape[0] != 1):
        trophic = trophic[trophic[:,1] != 0]
    if(infected_prey.shape[0] != 1):
        infected_prey = infected_prey[infected_prey[:,1] != 0]
    if(infected_predator_trophic.shape[0] != 1):
        infected_predator_trophic = infected_predator_trophic[infected_predator_trophic[:,1] != 0]
    if(infected_predator_active.shape[0] != 1):
        infected_predator_active = infected_predator_active[infected_predator_active[:,1] != 0]

    # Optimize arrays (show only those that are still alive or were effective)
    if(store_prey.shape[0] != 1):
        store_prey = store_prey[np.logical_not(np.logical_and(store_prey[:,1] == 0, store_prey[:,6] == 0))]
    if(store_predator.shape[0] != 1):
        store_predator = store_predator[np.logical_not(np.logical_and(store_predator[:,1] == 0, store_predator[:,6] == 0))]
    if(store_active.shape[0] != 1):
        store_active = store_active[np.logical_not(np.logical_and(store_active[:,1] == 0, store_active[:,6] == 0))]
    if(store_trophic.shape[0] != 1):
        store_trophic = store_trophic[np.logical_not(np.logical_and(store_trophic[:,1] == 0, store_trophic[:,6] == 0))]
        
###### events uninfected prey ######
    prey_growth = prey[:,1] * gx # prey reproduction
    prey_death = prey[:,1] * dx # prey intrinsic death
    prey_competition = prey[:,1] * (sum(prey[:,1]) + sum(infected_prey[:,1])) * (1 /pop_limit) # prey death due to competition

###### events infected prey ######
    infected_prey_growth = infected_prey[:,1] * gx * rx # infected prey reproduction
    infected_prey_death = infected_prey[:,1] * dx # infected prey intrinsic death
    infected_prey_competition = infected_prey[:,1] * (sum(infected_prey[:,1]) + sum(prey[:,1])) * (1 /pop_limit) # infected prey death due to competition

###### events free-living parasite ######
    infection_prey = np.zeros([trophic.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,trophic.shape[0]):
        for j in range(0,prey.shape[0]):
            infection_prey[i,j] = trophic[i,1] * prey[j,1] * infection_of_prey(prey[j,2],trophic[i,2]) * T # parasite infects prey
    trophic_death = trophic[:,1] * dz # parasite intrinsic death

    infection_predator = np.zeros([active.shape[0],predator.shape[0]], dtype = float) # storage array for event
    non_infection_predator = np.zeros([active.shape[0],predator.shape[0]], dtype = float) # storage array for event
    for i in range(0,active.shape[0]):
        for j in range(0,predator.shape[0]):
            infection_predator[i,j] = active[i,1] * predator[j,1] * infection_of_predator(predator[j,2],active[i,2]) * S # parasite infects prey
    active_death = active[:,1] * dz # parasite intrinsic death

###### events uninfected predator ######
    predator_growth = np.zeros([predator.shape[0],prey.shape[0]], dtype = float) # storage array for event
    predator_non_growth = np.zeros([predator.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,predator.shape[0]):
        for j in range(0,prey.shape[0]):
            predator_growth[i,j] = predator[i,1] * prey[j,1] * fy * ky # predator reproduces after feeding
            predator_non_growth[i,j] = predator[i,1] * prey[j,1] * fy * (1-ky) # predator does not reproduce after feeding

    predator_exposure_growth = np.zeros([predator.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    predator_exposure_non_growth = np.zeros([predator.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    predator_infection_growth = np.zeros([predator.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    predator_infection_non_growth = np.zeros([predator.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,predator.shape[0]):
        for j in range(0,infected_prey.shape[0]):
            predator_exposure_growth[i,j] = predator[i,1] * infected_prey[j,1] * fy * (1-infection_of_predator(predator[i,2],infected_prey[j,3])) * ky # predator exposed to parasite reproduces
            predator_exposure_non_growth[i,j] = predator[i,1] * infected_prey[j,1] * fy * (1-infection_of_predator(predator[i,2],infected_prey[j,3])) * (1 - ky) # predator exposed to parasite does not reproduce
            predator_infection_growth[i,j] = predator[i,1] * infected_prey[j,1] * fy * infection_of_predator(predator[i,2],infected_prey[j,3]) * rp1 * ky # predator infected by parasite reproduces
            predator_infection_non_growth[i,j] = predator[i,1] * infected_prey[j,1] * fy * infection_of_predator(predator[i,2],infected_prey[j,3]) * (1 - (rp1 * ky)) # predator infected by parasite does not reproduce
    predator_death = predator[:,1] * dy # predator intrinsic death
    
    # events infected predator
    transmission_predator = np.zeros([infected_predator_active.shape[0],predator.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_predator_active.shape[0]):
        for j in range(0,predator.shape[0]):
            transmission_predator[i,j] = infected_predator_active[i,1] * predator[j,1] * infection_of_predator(predator[j,2],infected_predator_active[i,3]) * E # parasite infects prey

    infected_predator_active_growth = np.zeros([infected_predator_active.shape[0],prey.shape[0]], dtype = float) # storage array for event
    infected_predator_active_non_growth = np.zeros([infected_predator_active.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_predator_active.shape[0]):
        for j in range(0,prey.shape[0]):
            infected_predator_active_growth[i,j] = infected_predator_active[i,1] * prey[j,1] * fy * rp2 * ky # infected predator reproduces after feeding
            infected_predator_active_non_growth[i,j] = infected_predator_active[i,1] * prey[j,1] * fy * (1 - (rp2 * ky)) # infected predator does not reproduce after feeding
    
    infected_predator_active_exposure_growth = np.zeros([infected_predator_active.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    infected_predator_active_exposure_non_growth = np.zeros([infected_predator_active.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_predator_active.shape[0]):
        for j in range(0,infected_prey.shape[0]):
            infected_predator_active_exposure_growth[i,j] = infected_predator_active[i,1] * infected_prey[j,1] * fy * rp2 * ky # infected predator exposed to the parasite reproduces
            infected_predator_active_exposure_non_growth[i,j] = infected_predator_active[i,1] * infected_prey[j,1] * fy * (1 - (rp2 * ky))  # infected predator exposed to the parasite does not reproduce
    infected_predator_active_death = infected_predator_active[:,1] * dy # infected predator intrinsic death

    # events infected predator
    infected_predator_trophic_growth = np.zeros([infected_predator_trophic.shape[0],prey.shape[0]], dtype = float) # storage array for event
    infected_predator_trophic_non_growth = np.zeros([infected_predator_trophic.shape[0],prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_predator_trophic.shape[0]):
        for j in range(0,prey.shape[0]):
            infected_predator_trophic_growth[i,j] = infected_predator_trophic[i,1] * prey[j,1] * fy * rp1 * ky # infected predator reproduces after feeding
            infected_predator_trophic_non_growth[i,j] = infected_predator_trophic[i,1] * prey[j,1] * fy * (1 - (rp1 * ky)) # infected predator does not reproduce after feeding
    
    infected_predator_trophic_exposure_growth = np.zeros([infected_predator_trophic.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    infected_predator_trophic_exposure_non_growth = np.zeros([infected_predator_trophic.shape[0],infected_prey.shape[0]], dtype = float) # storage array for event
    for i in range(0,infected_predator_trophic.shape[0]):
        for j in range(0,infected_prey.shape[0]):
            infected_predator_trophic_exposure_growth[i,j] = infected_predator_trophic[i,1] * infected_prey[j,1] * fy * rp1 * ky # infected predator exposed to the parasite reproduces
            infected_predator_trophic_exposure_non_growth[i,j] = infected_predator_trophic[i,1] * infected_prey[j,1] * fy * (1 - (rp1 * ky))  # infected predator exposed to the parasite does not reproduce
    infected_predator_trophic_death = infected_predator_trophic[:,1] * dy # infected predator intrinsic death
    
    # Sum all events
    sum_events = (prey_growth.sum() + prey_death.sum() + prey_competition.sum() + 
    infected_prey_growth.sum() + infected_prey_death.sum() + infected_prey_competition.sum() + 
    infection_prey.sum() + infection_predator.sum() +
    transmission_predator.sum() + active_death.sum() + trophic_death.sum() +
    predator_growth.sum() + predator_non_growth.sum() + 
    predator_exposure_growth.sum() + predator_exposure_non_growth.sum() + 
    predator_infection_growth.sum() + predator_infection_non_growth.sum() + predator_death.sum() + 
    infected_predator_active_growth.sum() + infected_predator_active_non_growth.sum() + 
    infected_predator_active_exposure_growth.sum() + infected_predator_active_exposure_non_growth.sum()  + 
    infected_predator_active_death.sum() +
    infected_predator_trophic_growth.sum() + infected_predator_trophic_non_growth.sum() + 
    infected_predator_trophic_exposure_growth.sum() + infected_predator_trophic_exposure_non_growth.sum()  + 
    infected_predator_trophic_death.sum())

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
        for i in range(0,infected_prey.shape[0]):
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
        for i in range(0,trophic.shape[0]):
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

        if occurrence:
            break

        for i in range(0,active.shape[0]):
            for j in range(0,predator.shape[0]):
                if URN > P and URN <= P + infection_predator[i,j]/sum_events:
                    bx = 2 # nothing happens to prey
                    bz = 5 # parasite gets in the predator and reproduces
                    by = 7 # predator gets infected
                    mby = j
                    mbz = i
                    occurrence = True
                    break
                P += infection_predator[i,j]/sum_events
        
        if occurrence:
            break

        for i in range(0,infected_predator_active.shape[0]):
            for j in range(0,predator.shape[0]):    
                if URN > P and URN <= P + transmission_predator[i,j]/sum_events:
                    bx = 2 # nothing happens to prey
                    bz = 3 # parasite reproduces
                    by = 9 # predator gets infected
                    mby = j
                    mbi = i
                    occurrence = True
                    break
                P += transmission_predator[i,j]/sum_events

        if occurrence:
            break

        for i in range(0,active.shape[0]):
            if URN > P and URN <= P + active_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 4 # free-living parasite decreases by one
                by = 2 # nothing happens to predator
                mbz = i
                occurrence = True
                break
            P += active_death[i]/sum_events
        
        if occurrence:
            break

        for i in range(0,trophic.shape[0]):
            if URN > P and URN <= P + trophic_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 0 # free-living parasite decreases by one
                by = 2 # nothing happens to predator
                mbz = i
                occurrence = True
                break
            P += trophic_death[i]/sum_events
        
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
            for j in range(0,infected_prey.shape[0]):
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

        ################ Infected predator trophic #####################
        for i in range(0,infected_predator_trophic.shape[0]):
            for j in range(0,prey.shape[0]):
                if URN > P and URN <= P + infected_predator_trophic_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 6 # predator increases by one
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += infected_predator_trophic_growth[i,j]/sum_events
                                
                if URN > P and URN <= P + infected_predator_trophic_non_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += infected_predator_trophic_non_growth[i,j]/sum_events
        
        if occurrence:
            break

        for i in range(0,infected_predator_trophic.shape[0]):
            for j in range(0,infected_prey.shape[0]):
                if URN > P and URN <= P + infected_predator_trophic_exposure_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 6 # predator increases by one
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_trophic_exposure_growth[i,j]/sum_events
                                    
                if URN > P and URN <= P + infected_predator_trophic_exposure_non_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_trophic_exposure_non_growth[i,j]/sum_events

        if occurrence:
            break

        for i in range(0,infected_predator_trophic.shape[0]):
            if URN > P and URN <= P + infected_predator_trophic_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 2 # nothing happens to free-living parasite
                by = 5 # infected predator decreases by one  
                mby = i
                occurrence = True
                break
            P += infected_predator_trophic_death[i]/sum_events
    
        if occurrence:
            break

        ################ Infected predator active #####################
        for i in range(0,infected_predator_active.shape[0]):
            for j in range(0,prey.shape[0]):
                if URN > P and URN <= P + infected_predator_active_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 8 # predator increases by one
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += infected_predator_active_growth[i,j]/sum_events
                                
                if URN > P and URN <= P + infected_predator_active_non_growth[i,j]/sum_events:
                    bx = 0 # ancestral prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbx = j
                    occurrence = True
                    break
                P += infected_predator_active_non_growth[i,j]/sum_events
        
        if occurrence:
            break

        for i in range(0,infected_predator_active.shape[0]):
            for j in range(0,infected_prey.shape[0]):
                if URN > P and URN <= P + infected_predator_active_exposure_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 8 # predator increases by one
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_active_exposure_growth[i,j]/sum_events
                                    
                if URN > P and URN <= P + infected_predator_active_exposure_non_growth[i,j]/sum_events:
                    bx = 5 # infected prey decreases by one
                    bz = 2 # nothing happens to free-living parasite
                    by = 2 # nothing happens to predator
                    mby = i
                    mbi = j
                    occurrence = True
                    break
                P += infected_predator_active_exposure_non_growth[i,j]/sum_events

        if occurrence:
            break

        for i in range(0,infected_predator_active.shape[0]):
            if URN > P and URN <= P + infected_predator_active_death[i]/sum_events:
                bx = 2 # nothing happens to prey
                bz = 2 # nothing happens to free-living parasite
                by = 10 # infected predator decreases by one  
                mby = i
                occurrence = True
                break
            P += infected_predator_active_death[i]/sum_events
    
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

    if(bx == 0): # prey dies
        prey[mbx,1] -= 1 # decrease one prey individual
        store_prey = loop_to_remove(store_prey, prey[mbx,2])
    
    if(bx == 3):# parasite infects prey
        prey[mbx,1] -= 1  # prey gets infected
        infected_prey = loop_infection(infected_prey, prey[mbx,2], trophic[mbz,2]) # either append new genotype infected or add individual if already exists

    if(bx == 5): # infected prey dies
        infected_prey[mbi,1] -= 1 # decrease one infected prey from that particular row (where event happened)
        store_prey = loop_to_remove(store_prey, infected_prey[mbi,2])

    if(bx == 6): # infected prey reproduces
        new_infected_prey = np.array(my_mutation_loci(n_loci, mx, infected_prey[mbi,2])) # new genotype that results after reproduction (may have a mutation or not)
        prey = loop_to_compare_array(prey, new_infected_prey) # either append new genotype or add individual if already exists (it is added to uninfected prey array beacuse parasite is not transmitted from parent to offspring)
        store_prey = loop_to_store_array(store_prey, new_infected_prey, Time)
        temporal_prey = str(new_infected_prey) # check if offspring was super-resistant
        if(temporal_prey == str(np.ones(n_loci)) and emergence_time_prey == max_time):
            emergence_time_prey = Time
            emergence_st_prey = True

################### PARASITE EVENTS #######################
    if(bz == 0): # free-living parasite dies
        trophic[mbz,1] -= 1
        store_trophic = loop_to_remove(store_trophic, trophic[mbz,2])
    
    if(bz == 1): # parasite reproduces trophically
        new_parasite = np.zeros(n_z_trophic)
        for i in range(0,n_z_trophic): # repeat this for all parasite offspring
            new_parasite = np.array(my_mutation_loci(n_loci, mz, infected_prey[mbi,3])) # check for mutations
            trophic = loop_to_compare_array(trophic, new_parasite) # add new genotype (may contain mutation or may not)
            store_trophic = loop_to_store_array(store_trophic, new_parasite, Time)
        # If all-resistant or all-infective genotypes emerged, record the id and emergence time
        temporal_parasite = str(new_parasite) # check if offspring was super-infective
        if(temporal_parasite == str(np.ones(n_loci)) and emergence_time_trophic == max_time):
            emergence_time_trophic = Time
            emergence_st_trophic = True

    if(bz == 3): # parasite reproduces actively
        new_parasite = np.zeros(n_z_active)
        for i in range(0,n_z_active): # repeat this for all parasite offspring
            new_parasite = np.array(my_mutation_loci(n_loci, mz, infected_predator_active[mbi,3])) # check for mutations
            active = loop_to_compare_array(active, new_parasite) # add new genotype (may contain mutation or may not)
            store_active = loop_to_store_array(store_active, new_parasite, Time)
        # If all-resistant or all-infective genotypes emerged, record the id and emergence time
        temporal_parasite = str(new_parasite) # check if offspring was super-infective
        if(temporal_parasite == str(np.ones(n_loci)) and emergence_time_active == max_time):
            emergence_time_active = Time
            emergence_st_active = True
    
    if(bz == 4): # free-living parasite dies
        active[mbz,1] -= 1
        store_active = loop_to_remove(store_active, active[mbz,2])

    if(bz == 5): # free-living parasite dies but also reproduces in the predator
        active[mbz,1] -= 1
        store_active = loop_to_remove(store_active, active[mbz,2])
        new_parasite = np.zeros(n_z_active)
        for i in range(0,n_z_active): # repeat this for all parasite offspring
            new_parasite = np.array(my_mutation_loci(n_loci, mz, active[mbz,2])) # check for mutations
            active = loop_to_compare_array(active, new_parasite) # add new genotype (may contain mutation or may not)
            store_active = loop_to_store_array(store_active, new_parasite, Time)
        # If all-resistant or all-infective genotypes emerged, record the id and emergence time
        temporal_parasite = str(new_parasite) # check if offspring was super-infective
        if(temporal_parasite == str(np.ones(n_loci)) and emergence_time_active == max_time):
            emergence_time_active = Time
            emergence_st_active = True

################### PREDATOR EVENTS ########################
    if(by == 1): # predator reproduces
        new_predator = np.array(my_mutation_loci(n_loci, my, predator[mby,2])) # check for mutation in predator offspring
        predator = loop_to_compare_array(predator, new_predator) # add new genotype (may contain mutation or may not)
        store_predator = loop_to_store_array(store_predator, new_predator, Time)
        temporal_predator = str(new_predator) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True

    if(by == 0): # predator dies
        predator[mby,1] -= 1 # uninfected predator of that genotype (row in uninfected predator array) decreases by one
        store_predator = loop_to_remove(store_predator, predator[mby,2])

    if(by == 3): # predator gets infected and reproduces
        # Infection part
        predator[mby,1] -= 1
        infected_predator_trophic = loop_infection(infected_predator_trophic, predator[mby,2], infected_prey[mbi,3]) 
        # Reproduction part
        new_predator = np.array(my_mutation_loci(n_loci, my, predator[mby,2])) # the predator is also reproducing, create new genotype (may or may not have mutation)
        predator = loop_to_compare_array(predator, new_predator) # add new genotype to the uninfected predator array (parasite is not transmitted from parent to offspring)
        store_predator = loop_to_store_array(store_predator, new_predator, Time)
        temporal_predator = str(new_predator) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True

    if(by == 4): # predator gets infected and does not reproduce
        # Infection part
        predator[mby,1] -= 1
        infected_predator_trophic = loop_infection(infected_predator_trophic, predator[mby,2], infected_prey[mbi,3])

    if(by == 5): # infected predator trophic dies
        infected_predator_trophic[mby,1] -= 1 # uninfected predator of that genotype (row in uninfected predator array) decreases by one
        store_predator = loop_to_remove(store_predator, infected_predator_trophic[mby,2])

    if(by == 6): # infected predator trophic reproduces
        # Reproduction part
        new_infected_pred = np.array(my_mutation_loci(n_loci, my, infected_predator_trophic[mby,2])) # check for mutations in genotype of offspring
        predator = loop_to_compare_array(predator, new_infected_pred) # add genotype to uninfected predator array
        store_predator = loop_to_store_array(store_predator, new_infected_pred, Time)
        temporal_predator = str(new_infected_pred) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True
    
    if(by == 7):# parasite infects predator directly
        predator[mby,1] -= 1  # prey gets infected
        infected_predator_active = loop_infection(infected_predator_active, predator[mby,2], active[mbz,2]) # either append new genotype infected or add individual if already exists
    
    if(by == 8): # infected predator active reproduces
        # Reproduction part
        new_infected_pred = np.array(my_mutation_loci(n_loci, my, infected_predator_active[mby,2])) # check for mutations in genotype of offspring
        predator = loop_to_compare_array(predator, new_infected_pred) # add genotype to uninfected predator array
        store_predator = loop_to_store_array(store_predator, new_infected_pred, Time)
        temporal_predator = str(new_infected_pred) # check if offspring was super-resistant
        if(temporal_predator == str(np.ones(n_loci)) and emergence_time_predator == max_time):
            emergence_time_predator = Time
            emergence_st_predator = True

    if(by == 9): # predator gets infected with active
        # Infection part
        predator[mby,1] -= 1
        infected_predator_active = loop_infection(infected_predator_active, predator[mby,2], infected_predator_active[mbi,3])

    if(by == 10): # infected predator active dies
        infected_predator_active[mby,1] -= 1 # uninfected predator of that genotype (row in uninfected predator array) decreases by one
        store_predator = loop_to_remove(store_predator, infected_predator_active[mby,2])

    # Advance a step in time
    Time += dt_next_event # continuous time simulation

# Record relevant genotypes (same percentages for the three entities)
# If they have a "1", they are effective genotypes
    for i in range(0, store_prey.shape[0]): # goes extinct
        if(store_prey[i,1] == 0 and store_prey[i,3] == 1):
            store_prey[i,3] = 0
            store_prey[i,5] = Time
            store_prey[i,6] = Time - store_prey[i,4]

    for i in range(0, store_predator.shape[0]): # goes extinct
        if(store_predator[i,1] == 0 and store_predator[i,3] == 1):
            store_predator[i,3] = 0
            store_predator[i,5] = Time
            store_predator[i,6] = Time - store_predator[i,4]

    for i in range(0, store_active.shape[0]): # goes extinct
        if(store_active[i,1] == 0 and store_active[i,3] == 1):
            store_active[i,3] = 0
            store_active[i,5] = Time
            store_active[i,6] = Time - store_active[i,4]

    for i in range(0, store_trophic.shape[0]): # goes extinct
        if(store_trophic[i,1] == 0 and store_trophic[i,3] == 1):
            store_trophic[i,3] = 0
            store_trophic[i,5] = Time
            store_trophic[i,6] = Time - store_trophic[i,4]

    for i in range(0, store_prey.shape[0]):
        if sum(store_prey[:,1]) >= 1: # becomes effective genotype
            if(store_prey[i,1] >= (sum(store_prey[:,1]) * 0.02) and store_prey[i,3] == 0):
                store_prey[i,3] = 1

    for i in range(0, store_predator.shape[0]):
        if sum(store_predator[:,1]) >= 1: # becomes effective genotype
            if(store_predator[i,1] >= (sum(store_predator[:,1]) * 0.02) and store_predator[i,3] == 0):
                store_predator[i,3] = 1

    for i in range(0, store_active.shape[0]):
        if sum(store_active[:,1]) >= 1: # becomes effective genotype
            if(store_active[i,1] >= (sum(store_active[:,1]) * 0.02) and store_active[i,3] == 0):
                store_active[i,3] = 1

    for i in range(0, store_trophic.shape[0]):
        if sum(store_trophic[:,1]) >= 1: # becomes effective genotype
            if(store_trophic[i,1] >= (sum(store_trophic[:,1]) * 0.02) and store_trophic[i,3] == 0):
                store_trophic[i,3] = 1

    # Update number of effective genotypes
    effective_prey_genotype = sum(store_prey[:,3]) # number of prey genotypes that are effective
    effective_predator_genotype = sum(store_predator[:,3]) # number of predator genotypes that are effective
    effective_active_genotype = sum(store_active[:,3]) # number of active parasite genotypes that are effective
    effective_trophic_genotype = sum(store_trophic[:,3]) # number of trophic parasite genotypes that are effective
    # Update population sizes
    # free-living individuals
    sum_mutants_prey = np.sum(prey[:,1]) # uninfected prey
    sum_mutants_predator = np.sum(predator[:,1]) # uninfected predator
    sum_mutants_active = np.sum(active[:,1]) # free-living parasites
    sum_mutants_trophic = np.sum(trophic[:,1]) # free-living parasites
    # infected hosts
    sum_infected_prey = np.sum(infected_prey[:,1]) # infected prey
    sum_infected_predator_active = np.sum(infected_predator_active[:,1]) # infected predator
    sum_infected_predator_trophic = np.sum(infected_predator_trophic[:,1]) # infected predator
    # total
    sum_prey = sum_mutants_prey + sum_infected_prey # all prey (uninfected and infected)
    sum_predator = sum_mutants_predator + sum_infected_predator_active + sum_infected_predator_trophic # all predator (uninfected and infected)
    sum_active = sum_mutants_active # all active parasites (free-living)
    sum_trophic = sum_mutants_trophic # all trophic parasites (free-living)
    sum_parasite = sum_active + sum_trophic # all parasites (free-living)
    # Save active parasites rel abudance
    if sum(store_active[:,1]) != 0:
        active_rel_abundance = sum_active/sum_parasite
    # Save trophic parasites rel abudance
    if sum(store_trophic[:,1]) != 0:
        trophic_rel_abundance = sum_trophic/sum_parasite

# if prey, parasite, and predator go extinct, stop simulations (also record which one went extinct and the time)
    if(sum_prey <= 0 and not extinction_prey):
        extinction_prey = True
        extinction_time_prey = Time

    if(sum_active <= 0 and not extinction_active):
        extinction_active = True
        extinction_time_active = Time

    if(sum_trophic <= 0 and not extinction_trophic):
        extinction_trophic = True
        extinction_time_trophic = Time

    if(sum_predator <= 0 and not extinction_predator):
        extinction_predator = True
        extinction_time_predator = Time

    if Time > n and not extinction_active and not extinction_trophic and not extinction_predator and not extinction_prey:
        if n > recording_time:
            store_sum_prey.append(sum_prey)
            store_sum_predator.append(sum_predator)
            store_sum_parasite.append(sum_parasite)
            store_sum_active_abundance.append(sum_active)
            store_sum_trophic_abundance.append(sum_trophic)
            store_active_rel_abundance.append(active_rel_abundance)
            store_trophic_rel_abundance.append(trophic_rel_abundance)
            store_effective_prey.append(effective_prey_genotype)
            store_effective_predator.append(effective_predator_genotype)
            store_effective_active.append(effective_active_genotype)
            store_effective_trophic.append(effective_trophic_genotype)
        n += 1
  
    if(sum_active <= 0 and sum_trophic <= 0 and sum_predator <= 0):
        break
        
# Simulation finishes
# Record coexistence/extinctions
if not extinction_prey and not extinction_active and not extinction_trophic and not extinction_predator:
    coexistence = "1"
    coexistence_predator_prey_trophic = "0"
    coexistence_predator_prey_active = "0"
    coexistence_predator_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_active and not extinction_trophic and not extinction_predator:
    coexistence = "0"
    coexistence_predator_prey_trophic = "1"
    coexistence_predator_prey_active = "0"
    coexistence_predator_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and not extinction_active and extinction_trophic and not extinction_predator:
    coexistence = "0"
    coexistence_predator_prey_trophic = "0"
    coexistence_predator_prey_active = "1"
    coexistence_predator_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_active and extinction_trophic and not extinction_predator:
    coexistence = "0"
    coexistence_predator_prey_trophic = "0"
    coexistence_predator_prey_active = "0"
    coexistence_predator_prey = "1"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_active and extinction_trophic and extinction_predator:
    coexistence = "0"
    coexistence_predator_prey_trophic = "0"
    coexistence_predator_prey_active = "0"
    coexistence_predator_prey = "0"
    coexistence_prey = "1"
    extinction = "0"
elif extinction_prey and extinction_active and extinction_trophic and extinction_predator:
    coexistence = "0"
    coexistence_predator_prey_trophic = "0"
    coexistence_predator_prey_active = "0"
    coexistence_predator_prey = "0"
    coexistence_prey = "0"
    extinction = "1"

# Record persistance parasites
if extinction_active == True and extinction_trophic == True:
    persistance_neither = "1"
else:
    persistance_neither = "0"
if extinction_active == False and extinction_trophic == False:
    persistance_both = "1"
else:
    persistance_both = "0"
if extinction_active == False and extinction_trophic == True:
    persistance_active = "1"
else:
    persistance_active = "0"
if extinction_active == True and extinction_trophic == False:
    persistance_trophic = "1"
else:
    persistance_trophic = "0"

# Save effective genotypes lifespans that survived
for i in range(0, store_prey.shape[0]): # goes extinct
    if(store_prey[i,1] != 0 and store_prey[i,3] == 1):
        store_prey[i,5] = Time
        store_prey[i,6] = Time - store_prey[i,4]

for i in range(0, store_predator.shape[0]): # goes extinct
    if(store_predator[i,1] != 0 and store_predator[i,3] == 1):
        store_predator[i,5] = Time
        store_predator[i,6] = Time - store_predator[i,4]

for i in range(0, store_active.shape[0]): # goes extinct
    if(store_active[i,1] != 0 and store_active[i,3] == 1):
        store_active[i,5] = Time
        store_active[i,6] = Time - store_active[i,4]

for i in range(0, store_trophic.shape[0]): # goes extinct
    if(store_trophic[i,1] != 0 and store_trophic[i,3] == 1):
        store_trophic[i,5] = Time
        store_trophic[i,6] = Time - store_trophic[i,4]

# Save lifespans
# Prey
for i in range(0,store_prey.shape[0]):
    if store_prey[i,6] >= max_time * 0.1:
        lifespan_prey.append(store_prey[i,6])

# Predator
for i in range(0,store_predator.shape[0]):
    if store_predator[i,6] >= max_time * 0.1:
        lifespan_predator.append(store_predator[i,6])

# Active parasite
for i in range(0,store_active.shape[0]):
    if store_active[i,6] >= max_time * 0.1:
        lifespan_active.append(store_active[i,6])

# Trophic parasite
for i in range(0,store_trophic.shape[0]):
    if store_trophic[i,6] >= max_time * 0.1:
        lifespan_trophic.append(store_trophic[i,6])
        
# Save abundance genotypes and lifespans
if store_sum_prey != []:
    av_sum_prey = np.sum(store_sum_prey[:])/len(store_sum_prey)
else:
    av_sum_prey = 0
if store_sum_predator != []:
    av_sum_predator = np.sum(store_sum_predator[:])/len(store_sum_predator)
else:
    av_sum_predator = 0
if store_sum_parasite != []:
    av_sum_parasite = np.sum(store_sum_parasite[:])/len(store_sum_parasite)
else:
    av_sum_parasite = 0
if store_sum_active_abundance != []:
    av_active_abundance = np.sum(store_sum_active_abundance[:])/len(store_sum_active_abundance)
else:
    av_active_abundance = 0
if store_sum_trophic_abundance != []:   
    av_trophic_abundance = np.sum(store_sum_trophic_abundance[:])/len(store_sum_trophic_abundance)
else:
    av_trophic_abundance = 0
if store_active_rel_abundance != []:
    av_active_rel_abundance = np.sum(store_active_rel_abundance[:])/len(store_active_rel_abundance)
else:
    av_active_rel_abundance = 0
if store_trophic_rel_abundance != []:
    av_trophic_rel_abundance = np.sum(store_trophic_rel_abundance[:])/len(store_trophic_rel_abundance)
else:
    av_trophic_rel_abundance = 0
if store_effective_prey != []:
    av_effective_prey = np.sum(store_effective_prey[:])/len(store_effective_prey)
else:
    av_effective_prey = 0
if store_effective_predator != []:
    av_effective_predator = np.sum(store_effective_predator[:])/len(store_effective_predator)
else:
    av_effective_predator = 0
if store_effective_active != []:
    av_effective_active = np.sum(store_effective_active[:])/len(store_effective_active)
else:
    av_effective_active = 0
if store_effective_trophic != []:
    av_effective_trophic = np.sum(store_effective_trophic[:])/len(store_effective_trophic)
else:
    av_effective_trophic = 0
if lifespan_prey != []:
    av_lifespan_prey = np.sum(lifespan_prey[:])/len(lifespan_prey)
else:
    av_lifespan_prey = 0
if lifespan_predator != []:
    av_lifespan_predator = np.sum(lifespan_predator[:])/len(lifespan_predator)
else:
    av_lifespan_predator = 0
if lifespan_active != []:
    av_lifespan_active = np.sum(lifespan_active[:])/len(lifespan_active)
else:
    av_lifespan_active = 0
if lifespan_trophic != []:
    av_lifespan_trophic = np.sum(lifespan_trophic[:])/len(lifespan_trophic)
else:
    av_lifespan_trophic = 0

# Save output
List = [str(rp1),str(rp2),str(rx),str(av_sum_prey),str(av_sum_predator),str(av_sum_parasite),str(av_active_abundance),str(av_trophic_abundance),str(av_active_rel_abundance),str(av_trophic_rel_abundance),str(av_effective_prey),str(av_effective_predator),str(av_effective_active),str(av_effective_trophic),str(av_lifespan_prey),str(av_lifespan_predator),str(av_lifespan_active),str(av_lifespan_trophic)]
List2 = [str(rp1),str(rp2),str(rx),str(persistance_both),str(persistance_active),str(persistance_trophic),str(persistance_neither),str(coexistence),str(coexistence_predator_prey_active),str(coexistence_predator_prey_trophic),str(coexistence_predator_prey),str(coexistence_prey),str(extinction)]

# Open our existing CSV file in append mode
# Create a file object for this file
if store_sum_prey != []:
    with open("output_active.csv", 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()

with open("coexistence_active.csv", 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(List2)
    f_object.close()