# initial population sizes
ancestral_prey = 800 # uninfected prey
ancestral_parasite = 1000 # free-living parasites
ancestral_predator = 100 # uninfected predator
infected_prey = 0 # at the beginning of the simulation, all predator are uninfected
infected_predator = 0 # at the beginning of the simulation, all predator are uninfected

# prey parameters 
gx = 2.0 # growth rate
dx = 0.1 # intrinsic death
rx = 1.0 # reproduction factor for parasite infection
pop_limit = 2000 # population limit

# parasite parameters
n_z = 6 # number of offspring per reproduction event
dz = 0.09 # intrinsic death

#predator parameters
fy = 0.01 # predation rate
ky = 0.2 # reproduction rate
rp = 1.0 # reproduction factor for parasite infection
re = 1.0 # reproduction factor for parasite exposure
dy = 1.0 # intrinsic death

# infection
S = 0.0005 # scaling factor prey-parasite
sigma_value_predator = 0.85
sigma_value_prey = 0.85

# genotypes
n_loci = 10 # total number of loci in genotypes
mx = 0.00009 # mutation rate prey
mz = 0.00006 # mutation rate parasite
my = 0.00005 # mutation rate predator

# max time
max_time = 1000