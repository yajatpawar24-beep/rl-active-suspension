# Quarter-car parameters - standard academic values (course project)
ms  = 250.0     # sprung mass kg
mu  = 50.0      # unsprung mass kg
ks  = 16000.0   # suspension spring N/m
c0  = 1000.0    # passive damping Ns/m
kt  = 190000.0  # tyre stiffness N/m

C_MIN = 200.0   # semi-active damper range Ns/m
C_MAX = 4000.0

# Simulation timing
DT      = 1e-3    # physics step (1 kHz)
DT_CTRL = 10e-3   # control period (100 Hz)
N_SUB   = 10      # physics substeps per control step
T_SIM   = 5.0     # evaluation duration s
T_TRAIN = 2.0     # training episode duration s

# Observation normalisation
S_REF = 0.012   # suspension travel m
V_REF = 0.20    # velocity m/s
D_REF = 0.005   # tyre deflection m
TRAVEL_LIMIT = 0.08  # rattle-space hard limit m

# Reward weights (v3 - W3 raised to restore road-holding constraint)
W2   = 0.05    # suspension travel
W3   = 0.15    # tyre deflection
W_DU = 0.10    # chatter suppression
W_LIM = 10.0   # travel-violation penalty

G           = 9.81
STATIC_LOAD = (ms + mu) * G
FA_MAX      = 2000.0

# Collect all params into a single dict for passing to core/ modules
CFG = dict(
    ms=ms, mu=mu, ks=ks, c0=c0, kt=kt,
    C_MIN=C_MIN, C_MAX=C_MAX,
    DT=DT, DT_CTRL=DT_CTRL, N_SUB=N_SUB, T_SIM=T_SIM, T_TRAIN=T_TRAIN,
    S_REF=S_REF, V_REF=V_REF, D_REF=D_REF, TRAVEL_LIMIT=TRAVEL_LIMIT,
    W2=W2, W3=W3, W_DU=W_DU, W_LIM=W_LIM,
    G=G, STATIC_LOAD=STATIC_LOAD, FA_MAX=FA_MAX,
)
