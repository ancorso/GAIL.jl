using DeepQLearning, POMDPModelTools
using BSON: @save, load
using StaticArrays
include("gail.jl")
include("lavaworld.jl")

# Setup the problem parameters
sz = (7,5)
# lava = [(3,1), (4,1), (5,1), (3,5), (4,5), (5,5)]
# lava2 = [(3,4), (4,4), (5,4), (3,5), (4,5), (5,5)]
# lava_penalty = -1.0
# goals = [(7,5)]
# goal_reward = 1.0
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
Qnet() = Chain(Dense(input_dim, 256, relu), Dense(256,64, relu), Dense(64, 4, sigmoid)) 
Dnet() = Chain(Dense(input_dim, 256, relu), Dense(256,64, relu), Dense(64, 4), softmax)
dqn_steps = 20000 # to learn an expert policy
gail_steps = 2000
expert_buffer_size = 256 
nda_buffer_size = 256
λ_nda = 0.9f0 # Constant for NDA. λ = 1 ignores the NDA trajectories
# λ_ent = 0.1f0


# Build the mdp
mdp = SimpleGridWorld(size = sz, tprob = 1.)
# mdp = SimpleGridWorld(size = sz, tprob = 1., rewards = lavaworld_rewards(lava, lava_penalty, goals, goal_reward))
# mdp2 = SimpleGridWorld(size = sz, tprob = 1., rewards = lavaworld_rewards(lava2, lava_penalty, goals, goal_reward))

# solve with DeepQLearning to get expert trajectories
# qnet = Chain(Dense(input_dim, 256, relu), Dense(256, 64, relu), Dense(64, 4, tanh)) 
# dqn_solver = DeepQLearningSolver(qnetwork = qnet,
#                                  exploration_policy = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1., stop=0.1, steps=dqn_steps/2)),
#                                  max_steps = dqn_steps,
#                                  target_update_freq = 2000,
#                                  batch_size = 128,
#                                  learning_rate = 1f-3,
#                                  logdir="log/dqn")
# dqn_policy = solve(dqn_solver, mdp)
dqn_net = DeepQLearning.create_dueling_network(Chain(Dense(input_dim, 256, relu), Dense(256, 64, relu), Dense(64, 4, tanh))) 
Flux.loadparams!(dqn_net, load("log/dqn/qnetwork.bson")[:qnetwork])
dqn_policy = ChainPolicy(dqn_net, mdp)

# Show some samples
s = rand(initialstate(mdp))
undiscounted_reward(simulate(HistoryRecorder(max_steps = 100), mdp, dqn_policy, s))
render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy = dqn_policy)

# Solve with GAIL and NDA-Gail
buffer_eps = 10000
expert_trajectories = gen_buffer(mdp, ChainPolicy(dqn_net, mdp), buffer_eps, desired_return = 1.0)
gail_net = train_GAIL!(mdp, Qnet(), Dnet(), expert_trajectories, 
                          logdir = "log/gail/",
                          epochs = 5000)

# Solve with NDA-GAIL
nda_trajectories = gen_buffer(mdp, RandomPolicy(mdp), buffer_eps, desired_return = -1., nonzero_transitions_only = true)
nda_gail_net = train_GAIL!(mdp, Qnet(), Dnet(), expert_trajectories, 
                              nda_buff = nda_trajectories, 
                              logdir = "log/nda-gail/", 
                              λ_nda = 0.5f0, 
                              epochs = gail_steps)

## Print some returns
expected_return(mdp, ChainPolicy(dqn_net, mdp))
expected_return(mdp, ChainPolicy(gail_net, mdp))
expected_return(mdp, ChainPolicy(nda_gail_net, mdp))

expected_return(mdp2, ChainPolicy(dqn_net, mdp2))
expected_return(mdp2, ChainPolicy(gail_net, mdp2))
expected_return(mdp2, ChainPolicy(nda_gail_net, mdp2))


## Make some plots
using Cairo, Fontconfig, Compose, ColorSchemes
set_default_graphic_size(35cm,10cm)
r = compose(context(0,0,1cm, 0cm), Compose.rectangle()) # spacer

# Plot on the training MDP
expert_occupancy = gen_occupancy(expert_trajectories, mdp)
c_expert = render(mdp, (s = GWPos(7,5),), color = s->reward(mdp,s) <0 ? -10. :  Float64(expert_occupancy[s]) / 2., policy = ChainPolicy(dqn_net, mdp))
c_gail = render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy = ChainPolicy(gail_net, mdp))
c_nda_gail = render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy = ChainPolicy(nda_gail_net,mdp))
hstack(c_expert, r, c_gail, r, c_nda_gail) |> SVG("images/mdp1.svg")

c_expert2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(dqn_net, mdp2))
c_gail2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(gail_net, mdp2))
c_nda_gail2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(nda_gail_net, mdp2))
hstack(c_expert2, r, c_gail2, r, c_nda_gail2) |> SVG("images/mdp2.svg")

nda_occupancy = gen_occupancy(nda_trajectories, mdp2)
c_expert = render(mdp2, (s = GWPos(7,5),), color = s-> Float64(nda_occupancy[s]) / 1.6, policy = ChainPolicy(dqn_net, mdp))

