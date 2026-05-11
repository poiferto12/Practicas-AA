using Random
Random.seed!(1234)

include("firmas.jl")
include("extractFeatures.jl")

"""
Ejecuta validación cruzada estratificada para kNN clásico usando modelCrossValidation,
con los parámetros y métricas requeridos por la memoria y el enunciado.

Argumentos:
    inputs :: Matrix  Matriz de características
    targets :: Vector Vector de etiquetas
    cvIndices :: Vector Índices de validación cruzada
    ks :: Vector Valores de k a probar
    kfolds :: Int Número de folds (por defecto 10)

Devuelve:
    results :: Dict con métricas medias y desviaciones para cada k
    best_k  :: Valor de k con mejor F1-score
"""
function knn_modelCrossValidation(inputs, targets, cvIndices; ks=[1, 3, 5, 7, 9, 11])
    results = Dict()
    best_f1 = -Inf
    best_k = ks[1]
    best_conf_matrix = nothing

    println("==================================================")
    println("Resultados kNN")
    println("==================================================")

    for k in ks
        modelType = :KNeighborsClassifier
        modelHyperparameters = Dict("n_neighbors" => k)

        res = modelCrossValidation(modelType, modelHyperparameters, (inputs, targets), cvIndices)
        results[k] = res

        if res[7][1] > best_f1   # F1 medio
            best_f1 = res[7][1]
            best_k = k
            best_conf_matrix = res[8]
        end
    end

    println("\nResumen validación cruzada kNN:")
    for k in ks
        r = results[k]
        println("k=$k: F1=$(round(r[7][1], digits=4)) ± $(round(r[7][2], digits=4)), " *
                "acc=$(round(r[1][1], digits=4)) ± $(round(r[1][2], digits=4)), " *
                "sens=$(round(r[3][1], digits=4)) ± $(round(r[3][2], digits=4)), " *
                "espec=$(round(r[4][1], digits=4)) ± $(round(r[4][2], digits=4)), " *
                "VPP=$(round(r[5][1], digits=4)) ± $(round(r[5][2], digits=4))")
    end

    println("\nMejor k según F1-score: ", best_k)
    
    all_indices = collect(1:size(inputs, 1))
    train_hyperparameters = Dict("n_neighbors" => best_k)
    train_metrics = modelCrossValidation(:KNeighborsClassifier, train_hyperparameters, (inputs, targets), all_indices)
    
    train_accuracy = train_metrics[1]
    train_f1 = train_metrics[7]
    train_conf_matrix = train_metrics[8]
    
    return results, best_k, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix
end

# Verifica si datos ya están cargados (por script_resultados.jl)
if !(@isdefined inputs) || !(@isdefined targets) || !(@isdefined cvIndices)
    inputs, targets = loadDataset("./dataset")
    cvIndices = crossvalidation(targets, 10)
end

results, best_k, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix = knn_modelCrossValidation(inputs, targets, cvIndices)