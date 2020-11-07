using Flux, POMDPs, POMDPSimulators

mutable struct ExperienceBuffer
    s::Array{Float32, 2}
    a::Array{Float32, 2}
    sp::Array{Float32, 2}
    r::Array{Float32, 2}
    done::Array{Float32, 2}
    elements::Int64
    next_ind::Int64
end

ExperienceBuffer(sdim, adim, N) = ExperienceBuffer(zeros(sdim, N), zeros(adim, N), zeros(sdim, N), zeros(1,N), zeros(1,N), 0, 1)

empty_like(b::ExperienceBuffer) = ExperienceBuffer(size(b.s,1), size(b.a, 1), size(b.s, 2))

Base.length(b::ExperienceBuffer) = b.elements

function Base.push!(b::ExperienceBuffer, s, a, r, sp, done, mdp)
    b.s[:, b.next_ind] .= convert_s(AbstractVector, s, mdp) 
    b.a[:, b.next_ind] .= Flux.onehot(a, actions(mdp))
    b.sp[:, b.next_ind] .= convert_s(AbstractVector, sp, mdp)
    b.r[1, b.next_ind] = r
    b.done[1, b.next_ind] = done
    b.elements = min(length(b.r), b.elements + 1)
    b.next_ind = mod1(b.next_ind + 1,  length(b.r))
end

function gen_buffer(mdp, pol, N; desired_return = nothing, max_tries = 100*N)
    s = rand(initialstate(mdp))
    odim, adim = length(convert_s(AbstractVector, s, mdp)), length(actions(mdp))
    b = ExperienceBuffer(odim, adim, N)
    i, eps = 0, 0
    while length(b) < N && i < max_tries
        h = simulate(HistoryRecorder(max_steps = 1000), mdp, pol)
        if isnothing(desired_return) || undiscounted_reward(h) ≈ desired_return
            eps += 1
            for (s, a, r, sp) in eachstep(h, "(s, a, r, sp)")  
                push!(b, s, a, r, sp, isterminal(mdp, sp), mdp)
            end
        end
        i += 1
    end
    println("eps: ", eps)
    println("Took $eps episodes to fill buffer of size $N, for an average of $(N/eps) steps per ep")
    @assert length(b) == N
    b
end

function sample(b::Union{ExperienceBuffer,Nothing}, N::Int64)
    isnothing(b) && return nothing
    ids = randperm(b.elements)[1:N]
    (s = b.s[:,ids], a = b.a[:,ids], sp = b.sp[:,ids], r = b.r[:,ids], done = b.done[:,ids])
end

struct ChainPolicy <: Policy
    qnet
    mdp
end

POMDPs.action(p::ChainPolicy, s) = actions(mdp)[argmax(p.qnet(convert_s(AbstractVector, s, p.mdp)))]

function gen_occupancy(buffer, mdp)
    occupancy = Dict(s => 0 for s in states(mdp))
    for i=1:length(buffer)
        s = convert_s(GWPos, buffer.s[:,i], mdp)
        occupancy[s] += 1
    end
    occupancy
end

expected_return(mdp, policy, eps = 1000, agg = undiscounted_reward) = mean([agg(simulate(HistoryRecorder(max_steps = 100), mdp, policy)) for _=1:eps])
    

