# aproxSVM.jl
# Implementación experimental de SVM usando modelCrossValidation de firmas.jl

using Random
Random.seed!(1234)

include("firmas.jl")
include("extractFeatures.jl")

function svm_modelCrossValidation(inputs, targets, cvIndices; kfolds=10)
    # Configuraciones sencillas para evitar bloqueos
    configs = [
    Dict("C" => 0.1,  "kernel" => "linear"),
    Dict("C" => 1.0,  "kernel" => "linear"),
    Dict("C" => 10.0, "kernel" => "linear"),
    Dict("C" => 100.0,"kernel" => "linear"),
    Dict("C" => 0.1,  "kernel" => "rbf"),
    Dict("C" => 1.0,  "kernel" => "rbf"),
    Dict("C" => 10.0, "kernel" => "rbf"),
    Dict("C" => 100.0,"kernel" => "rbf")
]

    results = Vector{Tuple{Dict{String,Any}, Any}}()
    best_f1 = -Inf
    best_config = configs[1]
    best_conf_matrix = nothing

    println("==================================================")
    println("Resultados SVM")
    println("==================================================")

    for config in configs
        println("\nProbando configuración: ", config)

        res = modelCrossValidation(:SVC, config, (inputs, targets), cvIndices)
        push!(results, (config, res))

        if res[7][1] > best_f1
            best_f1 = res[7][1]
            best_config = config
            best_conf_matrix = res[8]
        end

        println("F1 = ", round(res[7][1], digits=4), " ± ", round(res[7][2], digits=4))
    end

    println("\nResumen validación cruzada SVM:")
    for (config, r) in results
        println("kernel=$(config["kernel"]), C=$(config["C"]): " *
                "F1=$(round(r[7][1], digits=4)) ± $(round(r[7][2], digits=4)), " *
                "acc=$(round(r[1][1], digits=4)) ± $(round(r[1][2], digits=4)), " *
                "sens=$(round(r[3][1], digits=4)) ± $(round(r[3][2], digits=4)), " *
                "espec=$(round(r[4][1], digits=4)) ± $(round(r[4][2], digits=4)), " *
                "VPP=$(round(r[5][1], digits=4)) ± $(round(r[5][2], digits=4))")
    end

    println("\nMejor configuración según F1-score: ", best_config)
    return results, best_config, best_conf_matrix
end

# Ejecutar directamente al hacer include:
# Verifica si datos ya están cargados (por script_resultados.jl)
if !(@isdefined inputs) || !(@isdefined targets) || !(@isdefined cvIndices)
    inputs, targets = loadDataset("./dataset")
    cvIndices = crossvalidation(targets, 10)
end

results, best_config = svm_modelCrossValidation(inputs, targets, cvIndices)