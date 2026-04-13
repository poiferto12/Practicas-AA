using Random
Random.seed!(1234)

include("firmas.jl")
include("extractFeatures.jl")

function dt_modelCrossValidation(inputs, targets, cvIndices; depths=[2,4,6,8,10,12])
    
    results = Dict()
    best_f1 = -Inf
    best_depth = depths[1]
    best_conf_matrix = nothing

    println("==================================================")
    println("Resultados Árboles de Decisión")
    println("==================================================")

    for depth in depths
        modelType = :DecisionTreeClassifier
        hyperparameters = Dict("max_depth" => depth)

        res = modelCrossValidation(modelType, hyperparameters, (inputs, targets), cvIndices)
        results[depth] = res

        if res[7][1] > best_f1   # F1 medio
            best_f1 = res[7][1]
            best_depth = depth
            best_conf_matrix = res[8]
        end
    end

    println("\nResumen Árboles de Decisión:")
    for depth in depths
        r = results[depth]
        println("depth=$depth: " *
    "acc=$(round(r[1][1], digits=4))±$(round(r[1][2], digits=4)), " *
    "sens=$(round(r[3][1], digits=4))±$(round(r[3][2], digits=4)), " *
    "spec=$(round(r[4][1], digits=4))±$(round(r[4][2], digits=4)), " *
    "prec=$(round(r[5][1], digits=4))±$(round(r[5][2], digits=4)), " *
    "F1=$(round(r[7][1], digits=4))±$(round(r[7][2], digits=4))"
)
    end

    println("\nMejor profundidad: ", best_depth)
    
    # Calcular métricas de entrenamiento en todo el dataset
    all_indices = collect(1:size(inputs, 1))
    train_hyperparameters = Dict("max_depth" => best_depth)
    train_metrics = modelCrossValidation(:DecisionTreeClassifier, train_hyperparameters, (inputs, targets), all_indices)
    
    train_accuracy = train_metrics[1]
    train_f1 = train_metrics[7]
    train_conf_matrix = train_metrics[8]
    
    return results, best_depth, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix
end

# Ejecutar
if !(@isdefined inputs) || !(@isdefined targets) || !(@isdefined cvIndices)
    inputs, targets = loadDataset("./dataset")
    cvIndices = crossvalidation(targets, 10)
end

results, best_depth, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix = dt_modelCrossValidation(inputs, targets, cvIndices)