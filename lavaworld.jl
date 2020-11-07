using POMDPs, POMDPModels, Random

POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(transition(mdp, s, a )), r = reward(mdp, s, a))

function POMDPs.initialstate(mdp::SimpleGridWorld)
    # return Deterministic(GWPos(1,5))
    while true
        x, y = rand(1:mdp.size[1]), rand(1:mdp.size[2])
        !(GWPos(x,y) in mdp.terminate_from) && return Deterministic(GWPos(x,y))
    end
end 
            
function POMDPs.convert_s(::Type{V}, s::GWPos, mdp::SimpleGridWorld) where {V<:AbstractArray}
    svec = zeros(Float32, mdp.size..., 3)
    !isterminal(mdp, s) && (svec[s[1], s[2], 3] = 1.)
    for p in states(mdp)
        reward(mdp, p) < 0 && (svec[p[1], p[2], 2] = 1.)
        reward(mdp, p) > 0 && (svec[p[1], p[2], 1] = 1.)
    end
    svec[:]
end

POMDPs.convert_s(::Type{GWPos}, v::V, mdp::SimpleGridWorld) where {V<:AbstractArray} = GWPos(findfirst(reshape(v, mdp.size..., :)[:,:,3] .== 1.0).I)

lavaworld_rewards(lava, lava_penalty, goals, goal_reward) = merge(Dict(GWPos(p...) => lava_penalty for p in lava), Dict(GWPos(p...) => goal_reward for p in goals))
