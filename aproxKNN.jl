# aproxKNN.jl
# Implementación experimental de kNN clásico usando modelCrossValidation de firmas.jl

using Random
Random.seed!(1234)

include("firmas.jl")
include("extractFeatures.jl")

"""
Ejecuta validación cruzada estratificada para kNN clásico usando modelCrossValidation,
con los parámetros y métricas requeridos por la memoria y el enunciado.

Argumentos:
    datasetFolder :: String  Ruta a la carpeta de audios
    ks            :: Vector  Valores de k a probar
    kfolds        :: Int     Número de folds (por defecto 10)

Devuelve:
    results :: Dict con métricas medias y desviaciones para cada k
    best_k  :: Valor de k con mejor F1-score
"""
function knn_modelCrossValidation(datasetFolder; ks=[1, 3, 5, 7, 9, 11], kfolds=10)
    inputs, targets = loadDataset(datasetFolder)

    results = Dict()
    best_f1 = -Inf
    best_k = ks[1]

    cvIndices = crossvalidation(targets, kfolds)

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
    return results, best_k
end

# Ejemplo de uso:
# results, best_k = knn_modelCrossValidation("./audios"; ks=[1, 3, 5, 7, 9, 11], kfolds=10)