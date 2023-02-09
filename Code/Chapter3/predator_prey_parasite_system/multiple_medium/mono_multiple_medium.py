## Import relevant packages and modules ##
import pandas as pd, math, statistics, random, numpy.random as nura, numpy as np, array as arr, matplotlib.pyplot as plt, matplotlib.patches as mpatches, sys, getopt, time
from csv import writer

# initial population sizes
ancestral_prey = 800 # uninfected prey
ancestral_parasite = 1000 # free-living parasites
ancestral_predator = 100 # uninfected predator
infected_prey = 0 # at the beginning of the simulation, all prey are uninfected
infected_predator = 0 # at the beginning of the simulation, all predator are uninfected

# prey parameters 
gx = 2.0 # growth rate
dx = 0.1 # intrinsic death
pop_limit = 2000 # population limit

# parasite parameters
n_z = 6 # number of offspring per reproduction event
dz = 0.09 # intrinsic death

#predator parameters
fy = 0.01 # predation rate
ky = 0.2 # reproduction rate
re = 1.0
dy = 1.0 # intrinsic death

# genotypes
n_loci = 10 # total number of loci in genotypes
mx = 0.00002 # mutation rate prey
mz = 0.000006 # mutation rate parasite
my = 0.00005 # mutation rate predator

rp = float(sys.argv[1])
rx = float(sys.argv[2])

# invasion
invasion_Time_parasite = 200 # initial time of introduction of invasive genotypes
time_steps = 10 # introduce new genotype every x time steps (total number of introductions)
final_invasion_Time_parasite = 300 # time limit of introduction of invasive genotypes
invasion_Ind_parasite = 1000 # number of invader individuals
invasion_parasite = True
max_time = 1000
recording_time = 400

# infection
S = 0.0005 # scaling factor prey-parasite
sigma_value_prey = 0.85
sigma_value_predator = 0.85

## Function for appending new genotypes (simulations)
def loop_to_compare_array(the_array, new_genotype):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array
        if(np.array_equal(new_genotype, the_array[jj, 2])):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one (dead or alive) sum one individual
            return the_array
            break
        if jj == the_array.shape[0] - 1:
            the_array = np.append(the_array,[[jj+1, 1, new_genotype]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
            return the_array

## Function for appending new genotypes (storage)
def loop_to_store_array(the_array, new_genotype, initial_time):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array
        if(np.array_equal(new_genotype, the_array[jj, 2]) and the_array[jj, 1] != 0):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one (dead or alive) sum one individual
            return the_array
            break
        if jj == the_array.shape[0] - 1:
            the_array = np.append(the_array,[[jj+1, 1, new_genotype, 0, initial_time, 0, 0]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
            return the_array

## Function for appending new genotypes parasite (simulations)
def loop_to_compare_array_parasite(the_array, new_genotype, invasion_status, genotype_id):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array
        if(np.array_equal(new_genotype, the_array[jj, 2]) and the_array[jj, 4] == genotype_id):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one (dead or alive) sum one individual
            return the_array
            break
        if jj == the_array.shape[0] - 1:
            the_array = np.append(the_array,[[jj+1, 1, new_genotype, invasion_status, genotype_id]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
            return the_array

## Function for appending new genotypes parasite (storage)
def loop_to_store_array_parasite(the_array, new_genotype, initial_time, invasion_status, genotype_id):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array
        if(np.array_equal(new_genotype, the_array[jj, 2]) and the_array[jj, 12] == genotype_id and the_array[jj, 1] != 0):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one and it is alive, sum one individual
            return the_array
            break
        if(np.array_equal(new_genotype, the_array[jj, 2]) and the_array[jj, 12] == genotype_id and the_array[jj, 6] == 0 and the_array[jj, 1] == 0):
            the_array[jj, 1] += 1 # if the new genotype is the same as an existing one and it is dead but has not been recorded for lifespan, sum one individual
            return the_array
            break
        if jj == the_array.shape[0] - 1:
            the_array = np.append(the_array,[[jj+1, 1, new_genotype, 0, initial_time, 0, 0, 0, 0, 0, 0, invasion_status, genotype_id, 0, 0]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
            return the_array

## Function to remove individuals (storage)
def loop_to_remove(the_array, new_genotype):
    for jj in range(0, the_array.shape[0]): # Go through all the rows in array
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
def loop_infection(array_infected, host_genotype, parasite_genotype, invasion_status, genotype_id):
    for ii in range(0, array_infected.shape[0]): # Go through all the rows in array of infected prey
        if(np.array_equal(array_infected[ii, 2], host_genotype) and np.array_equal(array_infected[ii, 3], parasite_genotype) and np.array_equal(array_infected[ii, 5], genotype_id)):
            array_infected[ii, 1] += 1 # if we already have that prey genotype infected by that parasite genotype then just add one individual to that row
            return array_infected
            break
        if ii == array_infected.shape[0] - 1: # else add the new combination of interaction by adding a row with that prey and parasite genotype (one individual)
            array_infected = np.append(array_infected,[[ii+1, 1, host_genotype, parasite_genotype, invasion_status, genotype_id]], axis=0)
            return array_infected

def introduction_invader(n_loci, initial_array): # give number of loci in genotype, mutation rate, and genotype that is reproducing and may mutate
    sum_mutation = 5 # all the alleles that will be infective
    temporal_array = np.array(initial_array) # asign a temporal name to the genotype that may (or may not) mutate
    mut = np.random.choice(n_loci, sum_mutation, replace = False) # pick randomly those loci that will mutate, where sum_mutation is the number of loci mutating, replace = False to avoid using the same locus twice.
    for ii in range(0, sum_mutation): # for all the loci mutating check whether it is a 0 or a 1
        if(temporal_array[mut[ii]] == 0): # if it is a 0, change it to 1
            temporal_array[mut[ii]] = 1
    return temporal_array

# Infection parameters
index_to_add = 0 # index of the predator that got infected to keep track of genotype in array of infected individuals

# Genotypes length
initial_prey = np.zeros(n_loci) # initial prey genotype (all loci are initially susceptible, i.e. zeros)
initial_parasite = np.zeros(n_loci) # initial parasite genotype (all loci are initially non-infective, i.e. zeros)
initial_predator = np.zeros(n_loci) # initial predator genotype (all loci are initially susceptible, i.e. zeros)

# invasive genotypes
invasive_parasite = introduction_invader(n_loci, initial_parasite) # create new invader genotype
time_new_genotype = invasion_Time_parasite + time_steps # introduce new genotype every x time steps
total_invasions = 0 # total number of genotypes that invade in each time step
total_establishments = 0 # total number of genotypes that are established in each time step
inv_success = 0
genotype_id = 1
first_invasion = False

# Population arrays
np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning) # to avoid warning messages that result due to python version conflict
# Prey
prey = np.array([[0,ancestral_prey,initial_prey]], dtype = object) # array preys for all genotypes (simulation purposes)
store_prey = np.array([[0,ancestral_prey,initial_prey,1,0,0,0]], dtype = object) # array preys for all genotypes (storage purposes)
# Parasite
parasite = np.array([[0,ancestral_parasite,initial_parasite,0,0]], dtype = object) # array parasite for all genotypes (simulation purposes)
store_parasite = np.array([[0,ancestral_parasite,initial_parasite,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = object) # array parasite for all genotypes (storage purposes)
# Predator
predator = np.array([[0,ancestral_predator,initial_predator]], dtype = object) # array predator for all genotypes (simulation purposes)
store_predator = np.array([[0,ancestral_predator,initial_predator,1,0,0,0]], dtype = object) # array predator for all genotypes (storage purposes)
# Infected prey
infected = np.array([[0,infected_prey,initial_prey,initial_parasite,0,0]], dtype = object) # array infected prey (genotypes)
# Infected predator
infected_pred = np.array([[0,infected_predator,initial_predator,initial_parasite,0,0]], dtype = object) # array infected predator (genotypes)
mut = np.zeros([n_loci]) # helping array for randomly choosing a location in the genotype that mutates

# Variables for recording time of extinctions
extinction_prey = False
extinction_parasite = False
extinction_non_invaders = False
extinction_invaders = False
extinction_predator = False

# Variables for recording coexistence
persistance_parasite = "0"
persistance_both = "0"
persistance_invaders = "0"
persistance_non_invaders = "0"
coexistence = "0"
coexistence_predator_and_prey = "0"
coexistence_parasite_and_prey = "0"
coexistence_prey = "0"
extinction = "0"

# Variables for recording relevant genotypes
total_effective_prey_genotype = 1
total_effective_parasite_genotype = 1
total_effective_predator_genotype = 1
partial_time = [] # empty array for saving timing average seconds per iteration

store_sum_prey = []
store_sum_predator = []
store_sum_parasite = []
store_sum_inv_abundance = []
store_sum_non_inv_abundance = []
store_non_inv_rel_abundance = []
store_inv_rel_abundance = []
store_effective_prey = []
store_effective_predator = []
store_effective_parasite = []
store_effective_invader = []
store_effective_non_invader = []
lifespan_prey = []
lifespan_predator = []
lifespan_parasite = []
lifespan_invasive = []
lifespan_non_invasive = []
establishment_rate = []

# Abundance genotypes and lifespans
av_sum_prey = 0
av_sum_predator = 0
av_sum_parasite = 0
av_inv_abundance = 0
av_non_inv_abundance = 0
av_non_inv_rel_abundance = 0
av_inv_rel_abundance = 0
av_effective_prey = 0
av_effective_predator = 0
av_effective_parasite = 0
av_effective_invader = 0
av_effective_non_invader = 0
av_lifespan_prey = 0
av_lifespan_predator = 0
av_lifespan_parasite = 0
av_lifespan_invasive = 0
av_lifespan_non_invasive = 0
av_establishment_rate = 0
invasion_success = 0

# Continuous time
Time = 0 # total Gillespie time
dt_next_event = 0 # random time step after event occurs (following the Gillespie algorithm). This quantity is summed to the total time (continuos time simulation)
n = 0 # number of steps for recording time points across simulations

while Time < max_time: # SIMULATION STARTS: repeat simulation until reaching max time

    if(prey.shape[0] != 1): # remove extinct genotypes for speeding simulations
        prey = prey[prey[:,1] != 0]
    if(infected.shape[0] != 1):
        infected = infected[infected[:,1] != 0]
    if(parasite.shape[0] != 1):
        parasite = parasite[parasite[:,1] != 0]
    if(predator.shape[0] != 1):
        predator = predator[predator[:,1] != 0]
    if(infected_pred.shape[0] != 1):
        infected_pred = infected_pred[infected_pred[:,1] != 0]

###### introduce an invasive parasite genotype ######
### New genotype ###
    if Time >= invasion_Time_parasite and Time >= time_new_genotype and Time <= final_invasion_Time_parasite: # super-resistant parasite genotype emerges
        found_genotype = False
        while not found_genotype: # make sure the invader does not exist yet
            checks = 0
            invasive_parasite = introduction_invader(n_loci, initial_parasite) # create new invader genotype
            for i in range(0,store_parasite.shape[0]):
                if np.array_equal(invasive_parasite, store_parasite[i,2]):
                    checks += 1
            if checks == 0:
                found_genotype = True
        time_new_genotype += time_steps

### Array for simulations ###
    if Time >= invasion_Time_parasite and Time <= final_invasion_Time_parasite: # super-resistant parasite genotype emerges
        if int(sum(store_parasite[:,1]) * 0.01) <= 1:
            invasion_Ind_parasite = 1
        else:
            invasion_Ind_parasite = int(sum(store_parasite[:,1]) * 0.01)
        for jj in range(0, parasite.shape[0]): # Go through all the rows in array_parasite
            if(np.array_equal(invasive_parasite, parasite[jj, 2]) and genotype_id == parasite[jj, 4]):
                parasite[jj, 1] += invasion_Ind_parasite # if the new genotype is the same as an existing one (dead or alive) sum one individual
                break
            if jj == parasite.shape[0] - 1:
                parasite = np.append(parasite,[[jj+1, invasion_Ind_parasite, invasive_parasite, 1, genotype_id]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
### Array for storage ###
        for jj in range(0, store_parasite.shape[0]): # Go through all the rows in array_parasite
            if np.array_equal(invasive_parasite, store_parasite[jj, 2]) and genotype_id == store_parasite[jj, 12]:
                store_parasite[jj, 1] += invasion_Ind_parasite # if the new genotype is the same as an existing one (dead or alive) sum one individual
                break
            if jj == store_parasite.shape[0] - 1:
                store_parasite = np.append(store_parasite,[[jj+1, invasion_Ind_parasite, invasive_parasite, 0, Time, 0, 0, 0, 0, 0, 0, 1, genotype_id, 1, 0]], axis=0) # if the new genotype is different than any of the existing ones (dead or alive) add another row with one individual of that genotype
                total_invasions += 1 # record total number of genotypes that invade
        
        invasion_Time_parasite += 1 # let simulations run and introduce next genotype when time reaches one more time step

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

    elif(bx == 0): # prey dies
        prey[mbx,1] -= 1 # decrease one prey individual
        store_prey = loop_to_remove(store_prey, prey[mbx,2])
    
    elif(bx == 3):# parasite infects prey
        prey[mbx,1] -= 1  # prey gets infected
        infected = loop_infection(infected, prey[mbx,2], parasite[mbz,2], parasite[mbz,3], parasite[mbz,4]) # either append new genotype infected or add individual if already exists

    elif(bx == 5): # infected prey dies
        infected[mbi,1] -= 1 # decrease one infected prey from that particular row (where event happened)
        store_prey = loop_to_remove(store_prey, infected[mbi,2])

    elif(bx == 6): # infected prey reproduces
        new_infected_prey = np.array(my_mutation_loci(n_loci, mx, infected[mbi,2])) # new genotype that results after reproduction (may have a mutation or not)
        prey = loop_to_compare_array(prey, new_infected_prey) # either append new genotype or add individual if already exists (it is added to uninfected prey array beacuse parasite is not transmitted from parent to offspring)
        store_prey = loop_to_store_array(store_prey, new_infected_prey, Time)

################### PARASITE EVENTS #######################
    if(bz == 0): # free-living parasite dies
        parasite[mbz,1] -= 1
        store_parasite = loop_to_remove(store_parasite, parasite[mbz,2])
    
    elif(bz == 1): # parasite may or may not infect an uninfected predator
        new_parasite = np.zeros(n_z)
        for i in range(0,n_z): # repeat this for all parasite offspring
            new_parasite = np.array(my_mutation_loci(n_loci, mz, infected[mbi,3])) # check for mutations
            parasite = loop_to_compare_array_parasite(parasite, new_parasite, infected[mbi,4], infected[mbi,5]) # add new genotype (may contain mutation or may not)
            store_parasite = loop_to_store_array_parasite(store_parasite, new_parasite, Time, infected[mbi,4], infected[mbi,5])
    
################### PREDATOR EVENTS ########################
    if(by == 1): # predator reproduces
        new_predator = np.array(my_mutation_loci(n_loci, my, predator[mby,2])) # check for mutation in predator offspring
        predator = loop_to_compare_array(predator, new_predator) # add new genotype (may contain mutation or may not)
        store_predator = loop_to_store_array(store_predator, new_predator, Time)

    elif(by == 0): # predator dies
        predator[mby,1] -= 1 # uninfected predator of that genotype (row in uninfected predator array) decreases by one
        store_predator = loop_to_remove(store_predator, predator[mby,2])

    elif(by == 3): # predator gets infected and reproduces
        # Infection part
        predator[mby,1] -= 1
        infected_pred = loop_infection(infected_pred, predator[mby,2], infected[mbi,3], infected[mbi,4], infected[mbi,5])
        # Reproduction part
        new_predator = np.array(my_mutation_loci(n_loci, my, predator[mby,2])) # the predator is also reproducing, create new genotype (may or may not have mutation)
        predator = loop_to_compare_array(predator, new_predator) # add new genotype to the uninfected predator array (parasite is not transmitted from parent to offspring)
        store_predator = loop_to_store_array(store_predator, new_predator, Time)

    elif(by == 4): # predator gets infected and does not reproduce
        # Infection part
        predator[mby,1] -= 1
        infected_pred = loop_infection(infected_pred, predator[mby,2], infected[mbi,3], infected[mbi,4], infected[mbi,5])

    elif(by == 5): # infected predator dies
        infected_pred[mby,1] -= 1
        store_predator = loop_to_remove(store_predator, infected_pred[mby,2])
        
    elif(by == 6): # infected predator reproduces
        # Reproduction part
        new_infected_pred = np.array(my_mutation_loci(n_loci, my, infected_pred[mby,2])) # check for mutations in genotype of offspring
        predator = loop_to_compare_array(predator, new_infected_pred) # add genotype to uninfected predator array
        store_predator = loop_to_store_array(store_predator, new_infected_pred, Time)

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
            if(store_parasite[i,1] >= (sum(store_parasite[:,1]) * 0.05) and store_parasite[i,3] == 0):
                total_effective_parasite_genotype += 1
                store_parasite[i,3] = 1

    for i in range(0, store_predator.shape[0]):
        if sum(store_predator[:,1]) >= 1: # becomes effective genotype
            if(store_predator[i,1] >= (sum(store_predator[:,1]) * 0.02) and store_predator[i,3] == 0):
                total_effective_predator_genotype += 1
                store_predator[i,3] = 1
    
    # Record the number of individuals of parental invader and offspring by invader ID
    for x in range(1, genotype_id + 1):
        provisional = []
        for i in range(0, store_parasite.shape[0]):
            if store_parasite[i,12] == x:
                provisional.append(store_parasite[i,1])
        for i in range(0, store_parasite.shape[0]):
            if store_parasite[i,13] == 1 and store_parasite[i,12] == x:
                store_parasite[i,14] = sum(provisional[:])

    for i in range(0, store_parasite.shape[0]): # if sum of parental and offspring is more than 5% of total parasites add initial establishment time
        if store_parasite[i,14] >= (sum(store_parasite[:,1]) * 0.05) and store_parasite[i,7] == 0:
            store_parasite[i,7] = Time

    for i in range(0, store_parasite.shape[0]): # if all the individuals of ID died remove establishment time
        if store_parasite[i,14] == 0:
            store_parasite[i,7] = 0

    for i in range(0, store_parasite.shape[0]):
        # if the genotype established and has been around as effective for at least 10 time steps and this has not been recorded yet and it is an original invader and it is still around:
            if store_parasite[i,7] != 0 and Time >= int(store_parasite[i,7] + 10) and store_parasite[i,10] != 1 and store_parasite[i,13] == 1 and store_parasite[i,14] != 0:
                store_parasite[i,8] = Time # record establishment time
                store_parasite[i,9] = Time - store_parasite[i,4] # record establishment time
                store_parasite[i,10] = 1 # switch off genotype (record genotype as established)
                total_establishments += 1

    # Update population sizes
    # store abundance invaders per time step
    prov_abundance = []
    prov2_abundance = []
    non_inv_abundance = 0
    inv_abundance = 0
    non_inv_rel_abundance = 0
    inv_rel_abundance = 0
    # Record invasive parasite individuals
    for jj in range(0,store_parasite.shape[0]):
        if store_parasite[jj,11] == 1: # if the genotype is invasive and it still alive:
            prov_abundance.append(store_parasite[jj,1])
    if prov_abundance != []:
        inv_abundance = sum(prov_abundance[:])
    # Record non-invasive parasite individuals
    for jj in range(0,store_parasite.shape[0]):
        if store_parasite[jj,11] == 0: # if the genotype is not invasive and it still alive:
            prov2_abundance.append(store_parasite[jj,1])
    if prov2_abundance != []:
        non_inv_abundance = sum(prov2_abundance[:])
    # Save invaders rel abudance
    if sum(store_parasite[:,1]) != 0:
        non_inv_rel_abundance = non_inv_abundance/sum(store_parasite[:,1])
        inv_rel_abundance = inv_abundance/sum(store_parasite[:,1])
    # free-living individuals
    sum_mutants_prey = np.sum(prey[:,1]) # uninfected prey
    sum_mutants_parasite = np.sum(parasite[:,1]) # free-living parasites
    sum_mutants_predator = np.sum(predator[:,1]) # uninfected predator
    # infected hosts
    sum_infected_prey = np.sum(infected[:,1]) # infected prey
    sum_infected_predator = np.sum(infected_pred[:,1]) # infected predator
    # total
    sum_prey = sum_mutants_prey + sum_infected_prey # all prey (uninfected and infected)
    sum_parasite = sum_mutants_parasite # all parasites (free-living)
    sum_predator = sum_mutants_predator + sum_infected_predator # all predator (uninfected and infected)
    # store effective genotypes
    effective_prey_genotype = sum(store_prey[:,3]) # number of prey genotypes that are effective
    effective_predator_genotype = sum(store_predator[:,3]) # number of predator genotypes that are effective
    effective_parasite_genotype = sum(store_parasite[:,3]) # number of parasite genotypes that are effective
    effective_invader = []
    for i in range(0, store_parasite.shape[0]):
        if store_parasite[i,11] == 1:
            effective_invader.append(store_parasite[i,3])
    effective_non_invader = []
    for i in range(0, store_parasite.shape[0]):
        if store_parasite[i,11] == 0:
            effective_non_invader.append(store_parasite[i,3])
    effective_invader_genotype = sum(effective_invader[:]) # number of invader parasite genotypes that are effective
    effective_non_invader_genotype = sum(effective_non_invader[:]) # number of non-invader parasite genotypes that are effective

# if prey, parasite, and predator go extinct, stop simulations (also record which one went extinct and the time)
    if sum_prey <= 0 and not extinction_prey:
        extinction_prey = True

    if Time >= final_invasion_Time_parasite and sum_parasite <= 0 and not extinction_parasite:
        extinction_parasite = True

    if non_inv_abundance <= 0 and not extinction_non_invaders:
        extinction_non_invaders = True
    
    if Time >= final_invasion_Time_parasite and inv_abundance <= 0 and not extinction_invaders:
        extinction_invaders = True

    if sum_predator <= 0 and not extinction_predator:
        extinction_predator = True

    if Time > n and not extinction_parasite and not extinction_predator and not extinction_prey:
        if n > recording_time:
            store_sum_prey.append(sum_prey)
            store_sum_predator.append(sum_predator)
            store_sum_parasite.append(sum_parasite)
            store_sum_inv_abundance.append(inv_abundance)
            store_sum_non_inv_abundance.append(non_inv_abundance)
            store_non_inv_rel_abundance.append(non_inv_rel_abundance)
            store_inv_rel_abundance.append(inv_rel_abundance)
            store_effective_prey.append(effective_prey_genotype)
            store_effective_predator.append(effective_predator_genotype)
            store_effective_parasite.append(effective_parasite_genotype)
            store_effective_invader.append(effective_invader_genotype)
            store_effective_non_invader.append(effective_non_invader_genotype)
        n += 1

    if extinction_parasite and extinction_predator:
        break
        
# Simulation finishes
# Record coexistence/extinctions
if not extinction_prey and not extinction_parasite and not extinction_predator:
    coexistence = "1"
    coexistence_predator_and_prey = "0"
    coexistence_parasite_and_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_parasite and not extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "1"
    coexistence_parasite_and_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
elif not extinction_prey and extinction_parasite and extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_parasite_and_prey = "0"
    coexistence_prey = "1"
    extinction = "0"
elif not extinction_prey and not extinction_parasite and extinction_predator: 
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_parasite_and_prey = "1"
    coexistence_prey = "0"
    extinction = "0"
elif extinction_prey and extinction_parasite and extinction_predator: 
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_parasite_and_prey = "0"
    coexistence_prey = "0"
    extinction = "1"

# Record persistance parasites
if extinction_parasite == False:
    persistance_parasite = "1"
else:
    persistance_parasite = "0"
if extinction_non_invaders == False and extinction_invaders == False and extinction_parasite == False:
    persistance_both = "1"
else:
    persistance_both = "0"
if extinction_non_invaders == True and extinction_invaders == False and extinction_parasite == False:
    persistance_invaders = "1"
else:
    persistance_invaders = "0"
if extinction_non_invaders == False and extinction_invaders == True and extinction_parasite == False:
    persistance_non_invaders = "1"
else:
    persistance_non_invaders = "0"

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

# Save lifespans
# Prey
for i in range(0,store_prey.shape[0]):
    if store_prey[i,6] >= max_time * 0.1:
        lifespan_prey.append(store_prey[i,6])

# Predator
for i in range(0,store_predator.shape[0]):
    if store_predator[i,6] >= max_time * 0.1:
        lifespan_predator.append(store_predator[i,6])

# Parasite
for i in range(0,store_parasite.shape[0]):
    if store_parasite[i,6] >= max_time * 0.1:
        lifespan_parasite.append(store_parasite[i,6])

# Invader
for i in range(0,store_parasite.shape[0]):
    if store_parasite[i,11] == 1 and store_parasite[i,6] >= max_time * 0.1:
        lifespan_invasive.append(store_parasite[i,6])

# Non invader
for i in range(0,store_parasite.shape[0]):
    if store_parasite[i,11] == 0 and store_parasite[i,6] >= max_time * 0.1:
        lifespan_non_invasive.append(store_parasite[i,6])

# Establishment time invaders
for i in range(0,store_parasite.shape[0]):
    if store_parasite[i,9] != 0:
        establishment_rate.append(store_parasite[i,9])
        
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
if store_sum_inv_abundance != []:
    av_inv_abundance = np.sum(store_sum_inv_abundance[:])/len(store_sum_inv_abundance)
else:
    av_inv_abundance = 0
if store_sum_non_inv_abundance != []:   
    av_non_inv_abundance = np.sum(store_sum_non_inv_abundance[:])/len(store_sum_non_inv_abundance)
else:
    av_non_inv_abundance = 0
if store_non_inv_rel_abundance != []:
    av_non_inv_rel_abundance = np.sum(store_non_inv_rel_abundance[:])/len(store_non_inv_rel_abundance)
else:
    av_non_inv_rel_abundance = 0
if store_inv_rel_abundance != []:
    av_inv_rel_abundance = np.sum(store_inv_rel_abundance[:])/len(store_inv_rel_abundance)
else:
    av_inv_rel_abundance = 0
if store_effective_prey != []:
    av_effective_prey = np.sum(store_effective_prey[:])/len(store_effective_prey)
else:
    av_effective_prey = 0
if store_effective_predator != []:
    av_effective_predator = np.sum(store_effective_predator[:])/len(store_effective_predator)
else:
    av_effective_predator = 0
if store_effective_parasite != []:
    av_effective_parasite = np.sum(store_effective_parasite[:])/len(store_effective_parasite)
else:
    av_effective_parasite = 0
if store_effective_invader != []:
    av_effective_invader = np.sum(store_effective_invader[:])/len(store_effective_invader)
else:
    av_effective_invader = 0
if store_effective_non_invader != []:
    av_effective_non_invader = np.sum(store_effective_non_invader[:])/len(store_effective_non_invader)
else:
    av_effective_non_invader = 0
if lifespan_prey != []:
    av_lifespan_prey = np.sum(lifespan_prey[:])/len(lifespan_prey)
else:
    av_lifespan_prey = 0
if lifespan_predator != []:
    av_lifespan_predator = np.sum(lifespan_predator[:])/len(lifespan_predator)
else:
    av_lifespan_predator = 0
if lifespan_parasite != []:
    av_lifespan_parasite = np.sum(lifespan_parasite[:])/len(lifespan_parasite)
else:
    av_lifespan_parasite = 0
if lifespan_invasive != []:
    av_lifespan_invasive = np.sum(lifespan_invasive[:])/len(lifespan_invasive)
else:
    av_lifespan_invasive = 0
if lifespan_non_invasive != []:
    av_lifespan_non_invasive = np.sum(lifespan_non_invasive[:])/len(lifespan_non_invasive)
else:
    av_lifespan_non_invasive = 0

# Save invaders success
if establishment_rate != []:
    av_establishment_rate = np.sum(establishment_rate[:])/len(establishment_rate)
else:
    establishment_rate = 0
invasion_success = total_establishments/total_invasions

# Save output
List = ["mono_multiple_medium",str(rp),str(rx),str(av_sum_prey),str(av_sum_predator),str(av_sum_parasite),str(av_non_inv_abundance),str(av_inv_abundance),str(av_non_inv_rel_abundance),str(av_inv_rel_abundance),str(av_effective_prey),str(av_effective_predator),str(av_effective_parasite),str(av_effective_non_invader),str(av_effective_invader),str(av_lifespan_prey),str(av_lifespan_predator),str(av_lifespan_parasite),str(av_lifespan_non_invasive),str(av_lifespan_invasive),str(av_establishment_rate),str(invasion_success)]
List2 = ["mono_multiple_medium",str(rp),str(rx),str(persistance_parasite),str(persistance_non_invaders),str(persistance_invaders),str(persistance_both),str(coexistence),str(coexistence_predator_and_prey),str(coexistence_parasite_and_prey),str(coexistence_prey),str(extinction)]

# Open our existing CSV file in append mode
# Create a file object for this file
if store_sum_prey != []:
    with open("output_mono_multiple_medium.csv", 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()

with open("coexistence_mono_multiple_medium.csv", 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(List2)
    f_object.close()