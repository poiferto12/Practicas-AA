using Random
using Statistics
using Flux
using Flux: onehotbatch, onecold, crossentropy
using WAV
using FFTW
using Printf
using Dates

include("cnnArchitectures.jl")

# ================================================================
# CONFIGURACIÓN
# ================================================================

Random.seed!(1234)

const DATASET_PATH = "dataset"

const K_FOLDS = 10
const MAX_EPOCHS = 200
const EARLY_STOPPING_PATIENCE = 30
const BATCH_SIZE = 16
const LEARNING_RATE = 1e-3

const TARGET_LENGTH = 4096
const RANDOM_SEED = 1234

# ================================================================
# CLASES
# ================================================================

classes = sort(filter(
    x -> isdir(joinpath(DATASET_PATH, x)),
    readdir(DATASET_PATH)
))

numClasses = length(classes)

println("Clases detectadas:")
println(classes)

if numClasses == 0
    error("No se han detectado clases. Revisa que exista la carpeta '$DATASET_PATH' y que dentro tenga una carpeta por clase.")
end

# ================================================================
# FFT 1D
# ================================================================

function audioToFFT(path::String)

    audio, fs = wavread(path)

    if ndims(audio) > 1
        audio = vec(mean(audio, dims=2))
    end

    audio = Float32.(audio)

    # Transformada de Fourier
    fftSignal = abs.(fft(audio))

    # Nos quedamos con la mitad positiva del espectro
    fftSignal = fftSignal[1:div(length(fftSignal), 2)]

    # Escala logarítmica para reducir diferencias de magnitud
    fftSignal .= log.(fftSignal .+ 1f-6)

    # Normalización min-max por muestra
    fftSignal .-= minimum(fftSignal)

    maxv = maximum(fftSignal)

    if maxv > 0
        fftSignal ./= maxv
    end

    # Longitud fija
    if length(fftSignal) > TARGET_LENGTH

        fftSignal = fftSignal[1:TARGET_LENGTH]

    else

        padding = TARGET_LENGTH - length(fftSignal)

        fftSignal = vcat(
            fftSignal,
            zeros(Float32, padding)
        )

    end

    # Formato para CNN 2D:
    # alto x ancho x canales x batch
    # aquí cada muestra queda como TARGET_LENGTH x 1 x 1
    return reshape(
        Float32.(fftSignal),
        TARGET_LENGTH,
        1,
        1
    )

end

# ================================================================
# CARGA DEL DATASET
# ================================================================

function loadDataset()

    xs = Array{Float32,3}[]
    y = String[]

    total = 0

    for className in classes

        folder = joinpath(DATASET_PATH, className)

        files = filter(
            f -> endswith(lowercase(f), ".wav"),
            readdir(folder)
        )

        println("Clase $className -> $(length(files)) archivos")

        for file in files

            path = joinpath(folder, file)

            try

                signal = audioToFFT(path)

                push!(xs, signal)
                push!(y, className)

                total += 1

            catch e

                println("Error leyendo: $path")
                println(e)

            end
        end
    end

    if total == 0
        error("No se ha cargado ningún audio. Revisa la ruta del dataset y los archivos .wav.")
    end

    X = Array{Float32,4}(undef, TARGET_LENGTH, 1, 1, total)

    for i in 1:total
        X[:,:,:,i] .= xs[i]
    end

    println("\nPatrones cargados: $total")

    return X, y

end

# ================================================================
# VALIDACIÓN CRUZADA ESTRATIFICADA
# ================================================================

function stratifiedKfolds(y::Vector{String}, classes::Vector{String}, k::Int)

    folds = [Int[] for _ in 1:k]

    for className in classes

        idx = findall(==(className), y)

        Random.shuffle!(idx)

        for (pos, sampleIdx) in enumerate(idx)

            foldNumber = mod1(pos, k)

            push!(folds[foldNumber], sampleIdx)

        end
    end

    allIdx = collect(1:length(y))

    result = []

    for i in 1:k

        testIdx = sort(folds[i])

        testSet = Set(testIdx)

        trainValIdx = [idx for idx in allIdx if !(idx in testSet)]

        push!(result, (trainValIdx, testIdx))

    end

    return result

end

# ================================================================
# SPLIT TRAIN / VALIDACIÓN DENTRO DE CADA FOLD
# ================================================================
#
# En cada fold:
#   - 10% queda como test por la validación cruzada.
#   - Del 90% restante se separa aproximadamente 1/9 para validación.
# Resultado aproximado:
#   - 80% entrenamiento
#   - 10% validación
#   - 10% test
# ================================================================

function trainValidationSplit(trainValIdx::Vector{Int}, y::Vector{String}, classes::Vector{String})

    trainIdx = Int[]
    valIdx = Int[]

    for className in classes

        classIdx = [idx for idx in trainValIdx if y[idx] == className]

        Random.shuffle!(classIdx)

        # Como trainValIdx es aproximadamente el 90%,
        # tomar 1/9 de este conjunto equivale a un 10% total.
        nVal = max(1, round(Int, length(classIdx) / 9))

        nVal = min(nVal, length(classIdx) - 1)

        if nVal <= 0
            append!(trainIdx, classIdx)
        else
            append!(valIdx, classIdx[1:nVal])
            append!(trainIdx, classIdx[nVal+1:end])
        end
    end

    Random.shuffle!(trainIdx)
    Random.shuffle!(valIdx)

    return trainIdx, valIdx

end

# ================================================================
# MATRIZ DE CONFUSIÓN
# ================================================================

function confusionMatrix(yTrue, yPred, classes)

    n = length(classes)

    mat = zeros(Float64, n, n)

    for (t, p) in zip(yTrue, yPred)

        i = findfirst(==(t), classes)
        j = findfirst(==(p), classes)

        if i !== nothing && j !== nothing
            mat[i,j] += 1
        end
    end

    return mat

end

# ================================================================
# MÉTRICAS
# ================================================================

function metricsFromConfusionMatrix(cm)

    total = sum(cm)

    if total == 0
        return (
            accuracy = 0.0,
            sensitivity = 0.0,
            specificity = 0.0,
            precision = 0.0,
            f1 = 0.0
        )
    end

    acc = sum(cm[i,i] for i in 1:size(cm,1)) / total

    precisions = Float64[]
    recalls = Float64[]
    f1s = Float64[]
    specs = Float64[]

    n = size(cm,1)

    for i in 1:n

        TP = cm[i,i]

        FP = sum(cm[:,i]) - TP
        FN = sum(cm[i,:]) - TP
        TN = total - TP - FP - FN

        precision = TP / max(TP + FP, 1)
        recall = TP / max(TP + FN, 1)
        specificity = TN / max(TN + FP, 1)

        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        push!(precisions, precision)
        push!(recalls, recall)
        push!(f1s, f1)
        push!(specs, specificity)

    end

    return (
        accuracy = acc,
        sensitivity = mean(recalls),
        specificity = mean(specs),
        precision = mean(precisions),
        f1 = mean(f1s)
    )

end

# ================================================================
# CREAR MINI-BATCHES
# ================================================================

function makeBatches(X, yOH; batchSize::Int=BATCH_SIZE)

    n = size(X, 4)

    idx = collect(1:n)

    Random.shuffle!(idx)

    batches = Vector{Tuple{Array{Float32,4}, Any}}()

    startIdx = 1

    while startIdx <= n

        endIdx = min(startIdx + batchSize - 1, n)

        batchIdx = idx[startIdx:endIdx]

        xb = X[:,:,:,batchIdx]
        yb = yOH[:,batchIdx]

        push!(batches, (xb, yb))

        startIdx = endIdx + 1

    end

    return batches

end

# ================================================================
# VALIDATION LOSS
# ================================================================

function validationLoss(model, Xval, yValOH)

    Flux.testmode!(model)

    lossValue = crossentropy(model(Xval), yValOH)

    Flux.trainmode!(model)

    return lossValue

end

# ================================================================
# ENTRENAMIENTO CNN CON VALIDACIÓN Y EARLY STOPPING
# ================================================================

function trainCNN(model, Xtrain, yTrainOH, Xval, yValOH)

    optState = Flux.setup(
        Adam(LEARNING_RATE),
        model
    )

    loss(model, x, y) = crossentropy(model(x), y)

    bestValLoss = Inf
    bestModel = deepcopy(model)
    bestEpoch = 0

    epochsWithoutImprovement = 0

    for epoch in 1:MAX_EPOCHS

        Flux.trainmode!(model)

        batches = makeBatches(
            Xtrain,
            yTrainOH;
            batchSize = BATCH_SIZE
        )

        for batch in batches

            Flux.train!(
                loss,
                model,
                [batch],
                optState
            )

        end

        valLoss = validationLoss(model, Xval, yValOH)

        if valLoss < bestValLoss - 1e-6

            bestValLoss = valLoss
            bestModel = deepcopy(model)
            bestEpoch = epoch
            epochsWithoutImprovement = 0

        else

            epochsWithoutImprovement += 1

        end

        if epoch % 10 == 0 || epoch == 1
            @printf(
                "   Época %3d/%d - val loss: %.6f - mejor: %.6f (época %d)\n",
                epoch,
                MAX_EPOCHS,
                valLoss,
                bestValLoss,
                bestEpoch
            )
        end

        if epochsWithoutImprovement >= EARLY_STOPPING_PATIENCE

            println(
                "   Early stopping en época $epoch. Mejor época: $bestEpoch"
            )

            break

        end
    end

    Flux.testmode!(bestModel)

    return bestModel, bestEpoch, bestValLoss

end

# ================================================================
# PREDICCIÓN
# ================================================================

function predictLabels(model, X)

    Flux.testmode!(model)

    preds = model(X)

    return onecold(preds, classes)

end

# ================================================================
# CROSS VALIDATION CNN
# ================================================================

function crossValidationCNN(modelBuilder, X, y)

    folds = stratifiedKfolds(
        y,
        classes,
        K_FOLDS
    )

    metricsList = []
    bestEpochs = Int[]
    valLosses = Float64[]

    globalCM = zeros(
        Float64,
        numClasses,
        numClasses
    )

    for (fold, (trainValIdx, testIdx)) in enumerate(folds)

        println("\nFold $fold / $K_FOLDS")

        trainIdx, valIdx = trainValidationSplit(
            trainValIdx,
            y,
            classes
        )

        println("   Train: $(length(trainIdx)) patrones")
        println("   Val:   $(length(valIdx)) patrones")
        println("   Test:  $(length(testIdx)) patrones")

        Xtrain = X[:,:,:,trainIdx]
        Xval   = X[:,:,:,valIdx]
        Xtest  = X[:,:,:,testIdx]

        yTrain = y[trainIdx]
        yVal   = y[valIdx]
        yTest  = y[testIdx]

        yTrainOH = onehotbatch(yTrain, classes)
        yValOH   = onehotbatch(yVal, classes)

        model = modelBuilder()

        bestModel, bestEpoch, bestValLoss = trainCNN(
            model,
            Xtrain,
            yTrainOH,
            Xval,
            yValOH
        )

        push!(bestEpochs, bestEpoch)
        push!(valLosses, bestValLoss)

        predLabels = predictLabels(
            bestModel,
            Xtest
        )

        cm = confusionMatrix(
            yTest,
            predLabels,
            classes
        )

        globalCM .+= cm

        metrics = metricsFromConfusionMatrix(cm)

        push!(metricsList, metrics)

        println(
            "   Accuracy test fold: ",
            round(metrics.accuracy, digits=4)
        )

        println(
            "   F1 test fold: ",
            round(metrics.f1, digits=4)
        )

    end

    accs  = [m.accuracy for m in metricsList]
    sens  = [m.sensitivity for m in metricsList]
    specs = [m.specificity for m in metricsList]
    precs = [m.precision for m in metricsList]
    f1s   = [m.f1 for m in metricsList]

    result = Dict(

        "accuracy_mean" => mean(accs),
        "accuracy_std"  => std(accs),

        "sens_mean" => mean(sens),
        "sens_std"  => std(sens),

        "spec_mean" => mean(specs),
        "spec_std"  => std(specs),

        "prec_mean" => mean(precs),
        "prec_std"  => std(precs),

        "f1_mean" => mean(f1s),
        "f1_std"  => std(f1s),

        "best_epoch_mean" => mean(bestEpochs),
        "best_epoch_std"  => std(bestEpochs),

        "val_loss_mean" => mean(valLosses),
        "val_loss_std"  => std(valLosses),

        "cm_total" => globalCM,
        "cm_avg" => globalCM ./ K_FOLDS

    )

    return result

end

# ================================================================
# ENTRENAMIENTO FINAL DEL MEJOR MODELO SOBRE TODO EL DATASET
# ================================================================

function trainBestOnFullDataset(modelBuilder, X, y)

    allIdx = collect(1:length(y))

    trainIdx, valIdx = trainValidationSplit(
        allIdx,
        y,
        classes
    )

    Xtrain = X[:,:,:,trainIdx]
    Xval   = X[:,:,:,valIdx]

    yTrain = y[trainIdx]
    yVal   = y[valIdx]

    yTrainOH = onehotbatch(yTrain, classes)
    yValOH   = onehotbatch(yVal, classes)

    model = modelBuilder()

    bestModel, bestEpoch, bestValLoss = trainCNN(
        model,
        Xtrain,
        yTrainOH,
        Xval,
        yValOH
    )

    predLabels = predictLabels(bestModel, X)

    cm = confusionMatrix(
        y,
        predLabels,
        classes
    )

    metrics = metricsFromConfusionMatrix(cm)

    return Dict(
        "accuracy" => metrics.accuracy,
        "sensitivity" => metrics.sensitivity,
        "specificity" => metrics.specificity,
        "precision" => metrics.precision,
        "f1" => metrics.f1,
        "best_epoch" => bestEpoch,
        "val_loss" => bestValLoss,
        "cm" => cm
    )

end

# ================================================================
# IMPRIMIR MATRIZ DE CONFUSIÓN EN TXT
# ================================================================

function printConfusionMatrix(io, cm, classes)

    header = @sprintf("%14s", "")

    for c in classes

        cname = uppercase(first(c, min(5, length(c))))

        header *= @sprintf("%12s", "Pred:$cname")

    end

    println(io, header)

    for i in 1:length(classes)

        row = @sprintf("%12s", "Real:" * classes[i])

        for j in 1:length(classes)

            row *= @sprintf(
                "%12.1f",
                cm[i,j]
            )

        end

        println(io, row)

    end

end

# ================================================================
# GUARDAR REPORTE TXT
# ================================================================

function saveReport(results, bestTrainingMetrics=nothing)

    filename = "reporteCNN.txt"

    io = open(filename, "w")

    println(io,
"╔════════════════════════════════════════════════════════════╗")
    println(io,
"║     REPORTE DE RESULTADOS - CNN CLASIFICACIÓN AUDIO      ║")
    println(io,
"╚════════════════════════════════════════════════════════════╝")

    println(io)

    println(io, "Fecha de ejecución: $(Dates.now())")
    println(io, "Semilla aleatoria: $RANDOM_SEED")
    println(io, "Validación cruzada: $K_FOLDS folds")
    println(io, "División por fold: 80% entrenamiento, 10% validación, 10% test")
    println(io, "Batch size: $BATCH_SIZE")
    println(io, "Épocas máximas: $MAX_EPOCHS")
    println(io, "Learning rate: $LEARNING_RATE")
    println(io, "Early stopping: $EARLY_STOPPING_PATIENCE épocas sin mejora")
    println(io, "Longitud FFT: $TARGET_LENGTH")

    println(io)

    modelos = join([name for (name, res) in results], ", ")

    println(io, "Arquitecturas ejecutadas: $modelos")

    println(io)
    println(io,
"============================================================")
    println(io)

    bestF1 = -1.0
    bestName = ""

    for (name, res) in results

        if res["f1_mean"] > bestF1

            bestF1 = res["f1_mean"]
            bestName = name

        end
    end

    println(io,
"╔════════════════════════════════════════════════════════════╗")
    println(io,
"║  MODELO: REDES CONVOLUCIONALES 1D SOBRE FFT")
    println(io,
"╚════════════════════════════════════════════════════════════╝")

    println(io)

    println(io, "ARQUITECTURAS PROBADAS: $(length(results))")

    println(io)
    println(io,
"┌─ RESULTADOS DETALLADOS ─────────────────────────────────┐")
    println(io)

    for (i, (name, res)) in enumerate(results)

        println(io, "$i. Arquitectura: $name")

        @printf(io,
            "   Accuracy:      %.4f ± %.4f\n",
            res["accuracy_mean"],
            res["accuracy_std"])

        @printf(io,
            "   Sensibilidad:  %.4f ± %.4f\n",
            res["sens_mean"],
            res["sens_std"])

        @printf(io,
            "   Especificidad: %.4f ± %.4f\n",
            res["spec_mean"],
            res["spec_std"])

        @printf(io,
            "   VPP (Precision): %.4f ± %.4f\n",
            res["prec_mean"],
            res["prec_std"])

        @printf(io,
            "   F1-Score:      %.4f ± %.4f\n",
            res["f1_mean"],
            res["f1_std"])

        @printf(io,
            "   Mejor época:   %.2f ± %.2f\n",
            res["best_epoch_mean"],
            res["best_epoch_std"])

        @printf(io,
            "   Val loss:      %.6f ± %.6f\n",
            res["val_loss_mean"],
            res["val_loss_std"])

        println(io)

    end

    println(io,
"└──────────────────────────────────────────────────────────┘")

    println(io)

    bestResult = nothing

    for (name, res) in results

        if name == bestName
            bestResult = res
            break
        end
    end

    println(io,
"┌─ MEJOR ARQUITECTURA (por F1-score) ─────────────────────┐")
    println(io, "│ Configuración: $bestName")

    @printf(io,
        "│ Accuracy:      %.4f ± %.4f\n",
        bestResult["accuracy_mean"],
        bestResult["accuracy_std"])

    @printf(io,
        "│ Sensibilidad:  %.4f ± %.4f\n",
        bestResult["sens_mean"],
        bestResult["sens_std"])

    @printf(io,
        "│ Especificidad: %.4f ± %.4f\n",
        bestResult["spec_mean"],
        bestResult["spec_std"])

    @printf(io,
        "│ VPP:           %.4f ± %.4f\n",
        bestResult["prec_mean"],
        bestResult["prec_std"])

    @printf(io,
        "│ F1-Score:      %.4f ± %.4f\n",
        bestResult["f1_mean"],
        bestResult["f1_std"])

    println(io,
"└──────────────────────────────────────────────────────────┘")

    println(io)

    println(io,
"┌─ MATRIZ DE CONFUSIÓN (promedio 10-fold) ────────────────┐")

    printConfusionMatrix(
        io,
        bestResult["cm_avg"],
        classes
    )

    println(io,
"└──────────────────────────────────────────────────────────┘")

    println(io)

    println(io,
"┌─ MATRIZ DE CONFUSIÓN (acumulada 10-fold) ───────────────┐")

    printConfusionMatrix(
        io,
        bestResult["cm_total"],
        classes
    )

    println(io,
"└──────────────────────────────────────────────────────────┘")

    if bestTrainingMetrics !== nothing

        println(io)

        println(io,
"┌─ MÉTRICAS DE TRAINING (modelo final) ───────────────────┐")

        @printf(io,
            "│ Accuracy:      %.4f\n",
            bestTrainingMetrics["accuracy"])

        @printf(io,
            "│ Sensibilidad:  %.4f\n",
            bestTrainingMetrics["sensitivity"])

        @printf(io,
            "│ Especificidad: %.4f\n",
            bestTrainingMetrics["specificity"])

        @printf(io,
            "│ VPP:           %.4f\n",
            bestTrainingMetrics["precision"])

        @printf(io,
            "│ F1-Score:      %.4f\n",
            bestTrainingMetrics["f1"])

        println(io,
"└──────────────────────────────────────────────────────────┘")

        println(io)

        println(io,
"┌─ MATRIZ CONFUSIÓN TRAINING ─────────────────────────────┐")

        printConfusionMatrix(
            io,
            bestTrainingMetrics["cm"],
            classes
        )

        println(io,
"└──────────────────────────────────────────────────────────┘")

    end

    println(io)
    println(io,
"------------------------------------------------------------")
    println(io)

    println(io,
"════════════════════════════════════════════════════════════")
    println(io, "FIN DEL REPORTE")
    println(io,
"════════════════════════════════════════════════════════════")

    close(io)

    println("\nReporte guardado en: $filename")

end

# ================================================================
# MAIN
# ================================================================

println("\nCargando dataset...\n")

X, y = loadDataset()

println("\nDistribución de clases:")

for c in classes
    println("  $c: $(count(==(c), y)) muestras")
end

println("\n================================================")
println("INICIO DE ENTRENAMIENTO CNN")
println("================================================\n")

architectures = [

    ("CNN_1",  () -> buildCnn1(numClasses)),
    ("CNN_2",  () -> buildCnn2(numClasses)),
    ("CNN_3",  () -> buildCnn3(numClasses)),
    ("CNN_4",  () -> buildCnn4(numClasses)),
    ("CNN_5",  () -> buildCnn5(numClasses)),
    ("CNN_6",  () -> buildCnn6(numClasses)),
    ("CNN_7",  () -> buildCnn7(numClasses)),
    ("CNN_8",  () -> buildCnn8(numClasses)),
    ("CNN_9",  () -> buildCnn9(numClasses)),
    ("CNN_10", () -> buildCnn10(numClasses))

]

results = Vector{Tuple{String, Dict}}()

for (name, builder) in architectures

    println("\n========================================")
    println("Entrenando arquitectura: $name")
    println("========================================\n")

    result = crossValidationCNN(
        builder,
        X,
        y
    )

    push!(results, (name, result))

    println(
        "\nF1-score medio $name: ",
        round(result["f1_mean"], digits=4)
    )

end

# Buscar la mejor arquitectura por F1 medio
bestName = ""
bestBuilder = nothing
bestF1 = -1.0

for (name, res) in results

    if res["f1_mean"] > bestF1

        bestF1 = res["f1_mean"]
        bestName = name

    end
end

for (name, builder) in architectures

    if name == bestName
        bestBuilder = builder
        break
    end
end

println("\n================================================")
println("MEJOR ARQUITECTURA: $bestName")
println("F1 medio: $(round(bestF1, digits=4))")
println("================================================\n")

println("Entrenando modelo final de la mejor arquitectura...\n")

bestTrainingMetrics = trainBestOnFullDataset(
    bestBuilder,
    X,
    y
)

saveReport(
    results,
    bestTrainingMetrics
)

println("\nFIN.")