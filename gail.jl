include("utils.jl")
using POMDPPolicies, Flux, TensorBoardLogger, Logging
using Flux.Optimise: update!

entropy(vals) = -mean(sum(vals .* log.(vals), dims = 1))

function BCELoss(D, data, val::Float32, λ_ent)
    vals = D(data[:s])
    yh = sum(vals .* data[:a], dims = 1)
    # Flux.Losses.logitbinarycrossentropy(yh, val)
    Flux.Losses.binarycrossentropy(yh, val) #- λ_ent*entropy(vals)
end

function Lᴰ(D, expert, policy, nda, λ_nda, λ_ent)
    BCELoss(D, expert, 1.f0, λ_ent) + 
    λ_nda*BCELoss(D, policy, 0.f0, λ_ent) + 
    (1.f0 - λ_nda)*BCELoss(D, nda, 0.f0, λ_ent)
end

function train_discriminator!(D, optD, exp_data, pol_data, nda_data, λ_nda, λ_ent)
    θ = Flux.params(D)
    loss, back = Flux.pullback(() -> Lᴰ(D, exp_data, pol_data, nda_data, λ_nda, λ_ent), θ)
    update!(optD, θ, back(1f0))
    loss
end

function Lᴳ(Q, D, data, γ, maxQ, λ_ent)
    vals = Q(data[:s])
    avals = sum(vals .* data[:a], dims = 1) 
    target = sum(D(data[:s]) .* data[:a], dims = 1) .+ γ .* data[:done] .* maxQ
    Flux.Losses.huber_loss(avals, target, agg=mean) #- λ_ent*entropy(vals)
end

function train_Qnetwork!(Q, D, optQ, γ::Float32, pol_data, nda_data, λ_nda, λ_ent)
    pol_maxQ = maximum(Q(pol_data[:sp]), dims=1)
    # nda_maxQ = maximum(Q(nda_data[:sp]), dims=1)
    θ = Flux.params(Q)
    # loss, back = Flux.pullback(() -> λ_nda*Lᴳ(Q, D, pol_data, γ, pol_maxQ, λ_ent) + (1.f0 - λ_nda)*Lᴳ(Q, D, nda_data, γ, nda_maxQ, λ_ent), θ)
    loss, back = Flux.pullback(() -> Lᴳ(Q, D, pol_data, γ, pol_maxQ, λ_ent), θ)
    update!(optQ, θ, back(1f0))
    loss
end 


function train_GAIL!(mdp, Q, D, expert_buff::ExperienceBuffer; 
                    nda_buff::ExperienceBuffer = deepcopy(expert_buff), 
                    λ_nda::Float32 = 1.f0,
                    λ_ent::Float32 = 0.0f0,
                    epochs = 1000, 
                    optQ = ADAM(1e-3), 
                    optD = ADAM(1e-3),
                    buffer_eps = 10,
                    batch_size = 32, 
                    ϵ = LinearDecaySchedule(1.,0.1, epochs/2),
                    eval_freq = 10,
                    eval_eps = 100,
                    logdir = "log/",
                    verbose_freq = 10,
                    max_eval_steps = 100)
                    
    policy = ChainPolicy(Q, mdp)
    policy_buff = gen_buffer(mdp, RandomPolicy(mdp), buffer_eps)
    s, γ = rand(initialstate(mdp)) , Float32(discount(mdp))
    avgr = -Inf
    logger = TBLogger(logdir, tb_increment)
    with_logger(logger) do
        avgr = mean([simulate(RolloutSimulator(max_steps = max_eval_steps), mdp, policy) for _=1:eval_eps])
        log_value(logger, "eval_reward", avgr, step = 0)
    end
    
    for i=1:epochs
        a = rand() < ϵ(i) ? rand(actions(mdp)) : action(policy, s)
        sp, r = gen(mdp, s, a)
        done = isterminal(mdp, sp)
        push!(policy_buff, s, a, r, sp, done, mdp)
        s = done ? rand(initialstate(mdp)) : sp 
        
        loss_D = train_discriminator!(D, optD, sample(expert_buff, batch_size), sample(policy_buff, batch_size), sample(nda_buff, batch_size), λ_nda, λ_ent)
        loss_Q = train_Qnetwork!(Q, D, optQ, γ, sample(policy_buff, batch_size), sample(nda_buff, batch_size), λ_nda, λ_ent)
        
        with_logger(logger) do
            if i % eval_freq == 0
                avgr = mean([undiscounted_reward(simulate(HistoryRecorder(max_steps = max_eval_steps), mdp, ChainPolicy(Q, mdp))) for _=1:eval_eps])
                log_value(logger, "eval_reward", avgr, step = i)
            end
            log_value(logger, "loss_D", loss_D, step = i)
            log_value(logger, "loss_Q", loss_Q, step = i)
            log_value(logger, "eps", ϵ(i), step = i)
            # @save string(logger.logdir, "/last_Q.bson") Q
        end
        
        (i % verbose_freq == 0) && println("Epoch: $i, Discriminator loss: $loss_D, Qnet loss: $loss_Q, last avg return: $avgr")
    end 
    Q
end

