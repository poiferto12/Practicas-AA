using Random
Random.seed!(1234)

include("firmas.jl")
include("extractFeatures.jl")

function dome_modelCrossValidation(inputs, targets, cvIndices; nodes=[2,4,6,8,10,12,14,16])

    results = Dict()
    best_f1 = -Inf
    best_nodes = nodes[1]
    best_conf_matrix = nothing

    println("==================================================")
    println("Resultados DoME")
    println("==================================================")

    for n in nodes
        modelType = :DoME
        hyperparameters = Dict("maximumNodes" => n)

        res = modelCrossValidation(modelType, hyperparameters, (inputs, targets), cvIndices)
        results[n] = res

        if res[7][1] > best_f1
            best_f1 = res[7][1]
            best_nodes = n
            best_conf_matrix = res[8]
        end
    end

    println("\nResumen DoME:")
    for n in nodes
        r = results[n]
        println("nodes=$n: " *
    "acc=$(round(r[1][1], digits=4))±$(round(r[1][2], digits=4)), " *
    "sens=$(round(r[3][1], digits=4))±$(round(r[3][2], digits=4)), " *
    "spec=$(round(r[4][1], digits=4))±$(round(r[4][2], digits=4)), " *
    "prec=$(round(r[5][1], digits=4))±$(round(r[5][2], digits=4)), " *
    "F1=$(round(r[7][1], digits=4))±$(round(r[7][2], digits=4))")
    end

    println("\nMejor número de nodos: ", best_nodes)
    return results, best_nodes, best_conf_matrix
end

# Ejecutar
if !(@isdefined inputs) || !(@isdefined targets) || !(@isdefined cvIndices)
    inputs, targets = loadDataset("./dataset")
    cvIndices = crossvalidation(targets, 10)
end

results, best_nodes, best_conf_matrix = dome_modelCrossValidation(inputs, targets, cvIndices)