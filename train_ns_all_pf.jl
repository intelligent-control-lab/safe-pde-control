using Revise
using Burgers
using DataDeps, MAT, MLUtils
using NeuralOperators, Flux
using BSON
using DataDeps, MAT, MLUtils
using NeuralOperators, Flux
using CUDA, FluxTraining, BSON
import Flux: params
using BSON: @save, @load
using ProgressBars
using Zygote
using Optimisers, ParameterSchedulers

using Burgers
using FluxTraining




function my_get_data(file_path; n = 10000, Δsamples = 1, grid_size = div(201, Δsamples), T = Float32)
    file = matopen(file_path)
    
    x_data = T.(collect(read(file, "a")[1:n, 1:Δsamples:end]'))
    y_data = T.(collect(read(file, "u")[1:n, 1:Δsamples:end]'))
    safe_labels = T.(collect(read(file, "safe")[1:n, 1:Δsamples:end]'))
    pf_labels = T.(collect(read(file, "pf")[1:n, 1:Δsamples:end]'))
    close(file)

    x_loc_data = Array{T, 3}(undef, 2, grid_size, n)
    x_loc_data[1, :, :] .= reshape(repeat(LinRange(0, 0.2, grid_size), n), (grid_size, n))
    x_loc_data[2, :, :] .= x_data

    return x_loc_data, reshape(y_data, 1, :, n), safe_labels, pf_labels
end

function my_get_dataloader(; ratio::Float64 = 0.99, batchsize = 128,opt_only=false)
    if opt_only
        @assert 1==2
        𝐱1, 𝐲1, safe1, pf1 = my_get_data("data_opt_ns_abs_0.125.mat") # _new_10
    
        data_train1, data_test1 = splitobs((𝐱1, 𝐲1, safe1, pf1), at = ratio)
        𝐱2, 𝐲2, safe2, pf2 = my_get_data("data_opt_ns_abs_0.125.mat")
        
        data_train2, data_test2 = splitobs((𝐱2, 𝐲2, safe2, pf2), at = ratio)
        𝐱3, 𝐲3, safe3, pf3 = my_get_data("data_opt_ns_abs_0.125.mat")
        
        data_train3, data_test3 = splitobs((𝐱3, 𝐲3, safe3, pf3), at = ratio)
    else
        𝐱1, 𝐲1, safe1, pf1 = my_get_data("data_opt_ns_new_abs_0.12.mat") # 0.145 is 0init
        
        data_train1, data_test1 = splitobs((𝐱1, 𝐲1, safe1, pf1), at = ratio)
        𝐱2, 𝐲2, safe2, pf2 = my_get_data("data_ppo_ns_new_abs_0.12.mat")
        
        data_train2, data_test2 = splitobs((𝐱2, 𝐲2, safe2, pf2), at = ratio)
        𝐱3, 𝐲3, safe3, pf3 = my_get_data("data_sac_ns_new_abs_0.12.mat")
        
        data_train3, data_test3 = splitobs((𝐱3, 𝐲3, safe3, pf3), at = ratio)
    end

    data_train1_x_pf = data_train1[1][:,:,:]
    data_test1_x_pf = data_test1[1][:,:,:]
    data_train1_y_pf = data_train1[2][:,:,:]
    data_test1_y_pf = data_test1[2][:,:,:]
    data_train1_safe_pf = data_train1[3][:,:]
    data_test1_safe_pf = data_test1[3][:,:]

    data_train2_x_pf = data_train2[1][:,:,:]
    data_test2_x_pf = data_test2[1][:,:,:]
    data_train2_y_pf = data_train2[2][:,:,:]
    data_test2_y_pf = data_test2[2][:,:,:]
    data_train2_safe_pf = data_train2[3][:,:]
    data_test2_safe_pf = data_test2[3][:,:]

    data_train3_x_pf = data_train3[1][:,:,:]
    data_test3_x_pf = data_test3[1][:,:,:]
    data_train3_y_pf = data_train3[2][:,:,:]
    data_test3_y_pf = data_test3[2][:,:,:]
    data_train3_safe_pf = data_train3[3][:,:]
    data_test3_safe_pf = data_test3[3][:,:]




    data_train = (cat(cat(data_train1_x_pf, data_train2_x_pf, dims=3), data_train3_x_pf, dims=3), 
                    cat(cat(data_train1_y_pf, data_train2_y_pf, dims=3), data_train3_y_pf, dims=3), 
                    cat(cat(data_train1_safe_pf, data_train2_safe_pf, dims=2), data_train3_safe_pf, dims=2)) # omit the last pf tumple
    data_test = (cat(cat(data_test1_x_pf, data_test2_x_pf, dims=3), data_test3_x_pf, dims=3), 
                cat(cat(data_test1_y_pf, data_test2_y_pf, dims=3), data_test3_y_pf, dims=3), 
                cat(cat(data_test1_safe_pf, data_test2_safe_pf, dims=2), data_test3_safe_pf, dims=2)) # # omit the last pf tumple
    loader_train = DataLoader(data_train, batchsize = batchsize, shuffle = true)
    loader_test = DataLoader(data_test, batchsize = batchsize, shuffle = false)

    return loader_train, loader_test
end

function train(; cuda = true, η₀ = 1.0f-3, λ = 1.0f-4, epochs = 500)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    @show 1
    model = FourierNeuralOperator(ch = (2, 64, 64, 64, 64, 64, 128, 1), modes = (16,), 
                                  σ = gelu)
    data = my_get_dataloader(;opt_only=false)
    optimiser = Flux.Optimiser(Flux.Optimise.WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss

    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device))

    fit!(learner, epochs)
    model = learner.model |> cpu
    @save "model/ns_FNO_all.bson" model

    return learner
end

function train_MNO_dense(; cuda = true, η₀ = 1.0f-3, λ = 1.0f-4, epochs = 500)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = MarkovNeuralOperator(ch = (2, 64, 64, 64, 64, 64, 1), modes = (16,), 
                                  σ = gelu)
    data = my_get_dataloader(;opt_only=false)
    optimiser = Flux.Optimiser(Flux.Optimise.WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss

    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device))

    fit!(learner, epochs)
    model = learner.model |> cpu
    @save "model/ns_MNO_all.bson" model

    return learner
end

function train_opt_only(; cuda = true, η₀ = 1.0f-3, λ = 1.0f-4, epochs = 500)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    @show 1
    model = FourierNeuralOperator(ch = (2, 64, 64, 64, 64, 64, 128, 1), modes = (16,), 
                                  σ = gelu)
    data = my_get_dataloader(;opt_only=true)
    optimiser = Flux.Optimiser(Flux.Optimise.WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss

    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device))

    fit!(learner, epochs)
    model = learner.model |> cpu
    @save "model/ns_FNO_all_opt_only.bson" model

    return learner
end



train(epochs=100) 
train_MNO_dense(epochs=100)
# train_opt_only(epochs=100) 


