## Import relevant packages and modules ##
import pandas as pd, math, statistics, random, numpy.random as nura, numpy as np, array as arr, matplotlib.pyplot as plt, matplotlib.patches as mpatches, sys, getopt, time
from csv import writer

# initial population sizes
uninfected_prey = 800 # uninfected prey
parasite = 1000 # free-living parasites
uninfected_predator = 100 # uninfected predator
infected_prey = 0 # at the beginning of the simulation, all predator are uninfected
infected_predator = 0 # at the beginning of the simulation, all predator are uninfected
sum_prey = uninfected_prey + infected_prey
sum_predator = uninfected_predator + infected_predator
sum_parasite = parasite + infected_prey + infected_predator

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
re = 1.0 # reproduction factor for parasite exposure
dy = 1.0 # intrinsic death

# infection
S = 0.0005 # scaling factor prey-parasite
Qx = float(sys.argv[1])
Qy = float(sys.argv[2])
rp = 1

# max time
recording_time = 130 # time for recording the abundance of each entity
max_time = 150 # run algorithm until reaching max time

# Empty arrays for storing abundances of subtypes
store_uninfected_prey = []
store_infected_prey = []
store_uninfected_predator = []
store_infected_predator = []

# Variables for recording time of extinctions
extinction_prey = False
extinction_predator = False
extinction_parasite = False

coexistence = "0"
coexistence_predator_and_prey = "0"
coexistence_prey = "0"
extinction = "1"

grey = "0"
yellow = "0"
black = "0"
red = "0"
white = "1"
outcome = "0"

# Continuous time
Time = 0 # total Gillespie time
recording_time = 130 # time for recording the abundance of each entity
dt_next_event = 0 # random time step after event occurs (following the Gillespie algorithm). This quantity is summed to the total time (continuos time simulation)
n = 0 # number of steps for recording time points across simulations

while Time < max_time and (uninfected_prey + infected_prey > 0): # SIMULATION STARTS: repeat simulation until reaching max time
        
###### events uninfected prey ######
    prey_growth = uninfected_prey * gx # prey reproduction
    prey_death = uninfected_prey * dx # prey intrinsic death
    prey_competition = uninfected_prey * (uninfected_prey + infected_prey) * (1 /pop_limit) # prey death due to competition

###### events infected prey ######
    infected_prey_growth = infected_prey * gx # infected prey reproduction
    infected_prey_death = infected_prey * dx # infected prey intrinsic death
    infected_prey_competition = infected_prey * (infected_prey + uninfected_prey) * (1 /pop_limit) # infected prey death due to competition

###### events free-living parasite ######
    infection_prey = parasite * uninfected_prey * Qx * S # parasite infects prey
    non_infection_prey = parasite * uninfected_prey * (1-Qx) * S # parasite fails infecting prey
    parasite_death = parasite * dz # parasite intrinsic death

###### events uninfected predator ######
    predator_growth = uninfected_predator * uninfected_prey * fy * ky # predator reproduces after feeding
    predator_non_growth = uninfected_predator * uninfected_prey * fy * (1-ky) # predator does not reproduce after feeding

    predator_exposure_growth = uninfected_predator * infected_prey * fy * (1-Qy) * re * ky # predator exposed to parasite reproduces
    predator_exposure_non_growth = uninfected_predator * infected_prey * fy * (1-Qy) * (1 - (re * ky)) # predator exposed to parasite does not reproduce
    predator_infection_growth = uninfected_predator * infected_prey * fy * Qy * rp * ky # predator infected by parasite reproduces
    predator_infection_non_growth = uninfected_predator * infected_prey * fy * Qy * (1 - (rp * ky)) # predator infected by parasite does not reproduce
    predator_death = uninfected_predator * dy # predator intrinsic death
    
    # events infected predator
    infected_predator_growth = infected_predator * uninfected_prey * fy * rp * ky # infected predator reproduces after feeding
    infected_predator_non_growth = infected_predator * uninfected_prey * fy * (1 - (rp * ky)) # infected predator does not reproduce after feeding
    
    infected_predator_exposure_growth = infected_predator * infected_prey * fy * (1-Qy) * re * rp * ky # infected predator exposed to the parasite reproduces
    infected_predator_exposure_non_growth = infected_predator * infected_prey * fy * (1-Qy) * (1 - (re * rp * ky))  # infected predator exposed to the parasite does not reproduce
    infected_predator_infection_growth = infected_predator * infected_prey * fy * Qy * rp * rp * ky # infected predator infected by parasite reproduces
    infected_predator_infection_non_growth = infected_predator * infected_prey * fy * Qy * (1 - (rp * rp * ky)) # infected predator infected by parasite does not reproduce
    infected_predator_death = infected_predator * dy # infected predator intrinsic death
    
    # Sum all events
    sum_events = (prey_growth + prey_death + prey_competition + 
    infected_prey_growth + infected_prey_death + infected_prey_competition + 
    infection_prey + non_infection_prey + parasite_death + predator_growth + predator_non_growth + 
    predator_exposure_growth + predator_exposure_non_growth + 
    predator_infection_growth + predator_infection_non_growth + predator_death + 
    infected_predator_growth + infected_predator_non_growth + 
    infected_predator_exposure_growth + infected_predator_exposure_non_growth  + 
    infected_predator_infection_growth + infected_predator_infection_non_growth + infected_predator_death)

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
        if URN > P and URN <= P + prey_growth/sum_events:
            bx = 1 # uninfected prey increases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += prey_growth/sum_events #! use += to modify in place
        if occurrence:
            break
            
        if URN > P and URN <= P + prey_death/sum_events:
            bx = 0 # uninfected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += prey_death/sum_events
        if occurrence:
            break
    
        if URN > P and URN <= P + prey_competition/sum_events:
            bx = 0 # uninfected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += prey_competition/sum_events
        if occurrence:
            break

    ############### Infected prey #################
        if URN > P and URN <= P + infected_prey_growth/sum_events:
            bx = 6 # infected prey reproduces
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += infected_prey_growth/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + infected_prey_death/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += infected_prey_death/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + infected_prey_competition/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += infected_prey_competition/sum_events
        if occurrence:
            break
                                        
    ################ Free-living parasite ####################
        if URN > P and URN <= P + infection_prey/sum_events:
            bx = 3 # prey is carrying a parasite (now it is infected)
            bz = 0 # parasite gets in the prey
            by = 2 # nothing happens to predator
            occurrence = True
        P += infection_prey/sum_events
        if occurrence:
            break
        
        if URN > P and URN <= P + non_infection_prey/sum_events:
            bx = 2 # nothing happens to prey (it is not infected)
            bz = 2 #  nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += non_infection_prey/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + parasite_death/sum_events:
            bx = 2 # nothing happens to prey
            bz = 0 # free-living parasite decreases by one
            by = 2 # nothing happens to predator
            occurrence = True
        P += parasite_death/sum_events
        if occurrence:
            break

    ################ Uninfected predator #####################
        if URN > P and URN <= P + predator_growth/sum_events:
            bx = 0 # ancestral prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 1 # predator increases by one
            occurrence = True
        P += predator_growth/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + predator_non_growth/sum_events:
            bx = 0 # ancestral prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += predator_non_growth/sum_events
        if occurrence:
            break
        
        if URN > P and URN <= P + predator_death/sum_events:
            bx = 2 # nothing happens to prey
            bz = 2 # nothing happens to free-living parasite
            by = 0 # predator decreases by one
            occurrence = True
        P += predator_death/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + predator_exposure_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 1 # predator increases by one
            occurrence = True
        P += predator_exposure_growth/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + predator_exposure_non_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += predator_exposure_non_growth/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + predator_infection_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
            by = 3 # predator is infected and increases by one
            occurrence = True
        P += predator_infection_growth/sum_events
        if occurrence:
            break
                        
        if URN > P and URN <= P + predator_infection_non_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
            by = 4 # predator is infected and does not reproduce
            occurrence = True
        P += predator_infection_non_growth/sum_events
        if occurrence:
            break

    ################ Infected predator #####################
        if URN > P and URN <= P + infected_predator_growth/sum_events:
            bx = 0 # ancestral prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 6 # predator increases by one
            occurrence = True
        P += infected_predator_growth/sum_events
        if occurrence:
            break
                            
        if URN > P and URN <= P + infected_predator_non_growth/sum_events:
            bx = 0 # ancestral prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += infected_predator_non_growth/sum_events
        if occurrence:
            break
    
        if URN > P and URN <= P + infected_predator_exposure_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 6 # predator increases by one
            occurrence = True
        P += infected_predator_exposure_growth/sum_events
        if occurrence:
            break
                            
        if URN > P and URN <= P + infected_predator_exposure_non_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 2 # nothing happens to free-living parasite
            by = 2 # nothing happens to predator
            occurrence = True
        P += infected_predator_exposure_non_growth/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + infected_predator_infection_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
            by = 6 # predator increases by one
            occurrence = True
        P += infected_predator_infection_growth/sum_events
        if occurrence:
            break
                                
        if URN > P and URN <= P + infected_predator_infection_non_growth/sum_events:
            bx = 5 # infected prey decreases by one
            bz = 1 # free-living parasite increases by one (parasite inside the prey reproduces in predator)
            by = 2 # nothing happens to predator
            occurrence = True
        P += infected_predator_infection_non_growth/sum_events
        if occurrence:
            break

        if URN > P and URN <= P + infected_predator_death/sum_events:
            bx = 2 # nothing happens to prey
            bz = 2 # nothing happens to free-living parasite
            by = 5 # infected predator decreases by one
            occurrence = True
        P += infected_predator_death/sum_events
        if occurrence:
            break

##################### PREY EVENTS #########################
    if(bx == 1): # prey reproduces
        uninfected_prey += 1

    if(bx == 0): # prey dies
        uninfected_prey -= 1
    
    if(bx == 3):# parasite infects prey
        uninfected_prey -= 1
        infected_prey += 1

    if(bx == 5): # infected prey dies
        infected_prey -= 1

    if(bx == 6): # infected prey reproduces
        uninfected_prey += 1

################### PARASITE EVENTS #######################
    if(bz == 0): # free-living parasite dies
        parasite -= 1
    
    if(bz == 1): # parasite may or may not infect an uninfected predator
        parasite += n_z
    
################### PREDATOR EVENTS ########################
    if(by == 1): # predator reproduces
        uninfected_predator += 1

    if(by == 0): # predator dies
        uninfected_predator -= 1

    if(by == 3): # predator gets infected and reproduces
        infected_predator += 1

    if(by == 4): # predator gets infected and does not reproduce
        uninfected_predator -= 1
        infected_predator += 1

    if(by == 5): # infected predator dies
        infected_predator -= 1
        
    if(by == 6): # infected predator reproduces
        uninfected_predator += 1

    # Advance a step in time
    Time += dt_next_event # continuous time simulation
    
    # Sum uninfected and infected hosts
    sum_prey = uninfected_prey + infected_prey
    sum_predator = uninfected_predator + infected_predator
    sum_parasite = parasite + infected_prey + infected_predator

    # Record extinctions
    if(sum_prey <= 0):
        extinction_prey = True
    if(sum_predator <= 0):
        extinction_predator = True
    if(sum_parasite <= 0):
        extinction_parasite = True

    # Record abundance subtypes
    if Time > n and not extinction_parasite and not extinction_predator and not extinction_prey:
        if n > recording_time:
            store_uninfected_prey.append(uninfected_prey)
            store_infected_prey.append(infected_prey)
            store_uninfected_predator.append(uninfected_predator)
            store_infected_predator.append(infected_predator)
        n += 1

# Record av abundances of subtypes
if store_uninfected_prey != []:
    av_uninfected_prey = sum(store_uninfected_prey[:]) / len(store_uninfected_prey)
else:
    av_uninfected_prey = 0
if store_infected_prey != []:
    av_infected_prey = sum(store_infected_prey[:]) / len(store_infected_prey)
else:
    av_infected_prey = 0
if store_uninfected_predator != []:
    av_uninfected_predator = sum(store_uninfected_predator[:]) / len(store_uninfected_predator)
else:
    av_uninfected_predator = 0
if store_infected_predator != []:
    av_infected_predator = sum(store_infected_predator[:]) / len(store_infected_predator)
else:
    av_infected_predator = 0

# Record coexistence/extinctions
if(sum_prey <= 0):
    extinction_prey = True
if(sum_predator <= 0):
    extinction_predator = True
if(sum_parasite <= 0):
    extinction_parasite = True

if not extinction_prey and not extinction_parasite and not extinction_predator:
    coexistence = "1"
    coexistence_predator_and_prey = "0"
    coexistence_prey = "0"
    extinction = "0"
if not extinction_prey and extinction_parasite and not extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "1"
    coexistence_prey = "0"
    extinction = "0"
if not extinction_prey and extinction_parasite and extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_prey = "1"
    extinction = "0"
if extinction_prey and extinction_parasite and extinction_predator:
    coexistence = "0"
    coexistence_predator_and_prey = "0"
    coexistence_prey = "0"
    extinction = "1"

if av_uninfected_prey < av_infected_prey and av_uninfected_predator < av_infected_predator and coexistence == "1":
    grey = "1"
    yellow = "0"
    black = "0"
    red = "0"
    white = "0"

if av_uninfected_prey >= av_infected_prey and av_uninfected_predator < av_infected_predator and coexistence == "1":
    grey = "0"
    yellow = "1"
    black = "0"
    red = "0"
    white = "0"

if av_uninfected_prey < av_infected_prey and av_uninfected_predator >= av_infected_predator and coexistence == "1":
    grey = "0"
    yellow = "0"
    black = "1"
    red = "0"
    white = "0"

if av_uninfected_prey >= av_infected_prey and av_uninfected_predator >= av_infected_predator and coexistence == "1":
    grey = "0"
    yellow = "0"
    black = "0"
    red = "1"
    white = "0"

# Save output
List = [str(Qx),str(Qy),str(rp),str(coexistence_prey),str(coexistence_predator_and_prey),str(coexistence),str(extinction),str(grey),str(yellow),str(black),str(red),str(white)]

# Open our existing CSV file in append mode
# Create a file object for this file
with open("output_Qx_Qy_rp1.csv", 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(List)
    f_object.close()