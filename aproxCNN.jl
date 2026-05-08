
using Random
using Statistics
using Flux
using Flux: onehotbatch, onecold, crossentropy
using WAV
using FFTW
using MLUtils
using Printf
using Dates

include("cnnArchitectures.jl")

# ================================================================
# CONFIGURACIÓN
# ================================================================

Random.seed!(1234)

DATASET_PATH = "dataset"

K_FOLDS = 10

EPOCHS = 20

LEARNING_RATE = 1e-3

TARGET_LENGTH = 4096

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

# ================================================================
# FFT
# ================================================================

function audioToFFT(path::String)

    audio, fs = wavread(path)

    if ndims(audio) > 1
        audio = vec(mean(audio, dims=2))
    end

    audio = Float32.(audio)

    fftSignal = abs.(fft(audio))

    fftSignal = fftSignal[1:div(length(fftSignal),2)]

    fftSignal .= log.(fftSignal .+ 1f-6)

    fftSignal .-= minimum(fftSignal)

    maxv = maximum(fftSignal)

    if maxv > 0
        fftSignal ./= maxv
    end

    if length(fftSignal) > TARGET_LENGTH

        fftSignal = fftSignal[1:TARGET_LENGTH]

    else

        padding = TARGET_LENGTH - length(fftSignal)

        fftSignal = vcat(
            fftSignal,
            zeros(Float32, padding)
        )

    end

    return reshape(Float32.(fftSignal),
        TARGET_LENGTH,
        1,
        1)

end

# ================================================================
# LOAD DATASET
# ================================================================

function loadDataset()

    X = Array{Float32,4}(undef,
        TARGET_LENGTH,
        1,
        1,
        0)

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

                X = cat(X,
                    reshape(signal,
                        TARGET_LENGTH,
                        1,
                        1,
                        1),
                    dims=4)

                push!(y, className)

                total += 1

            catch e

                println("Error leyendo: $path")
                println(e)

            end
        end
    end

    println("\nPatrones cargados: $total")

    return X, y

end

# ================================================================
# CONFUSION MATRIX
# ================================================================

function confusionMatrix(yTrue, yPred, classes)

    n = length(classes)

    mat = zeros(Float64, n, n)

    for (t,p) in zip(yTrue, yPred)

        i = findfirst(==(t), classes)
        j = findfirst(==(p), classes)

        mat[i,j] += 1

    end

    return mat

end

# ================================================================
# METRICS
# ================================================================

function metricsFromConfusionMatrix(cm)

    total = sum(cm)

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

        precision = TP / max(TP+FP, 1)

        recall = TP / max(TP+FN, 1)

        specificity = TN / max(TN+FP, 1)

        f1 = 2 * precision * recall /
             max(precision + recall, 1e-8)

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
# TRAIN CNN
# ================================================================

function trainCNN(model, xTrain, yTrainOH)

    opt_state = Flux.setup(
        Adam(LEARNING_RATE),
        model
    )

    loss(model, x, y) =
        crossentropy(model(x), y)

    data = [(xTrain, yTrainOH)]

    for epoch in 1:EPOCHS

        Flux.train!(
            loss,
            model,
            data,
            opt_state
        )

    end

end

# ================================================================
# CROSS VALIDATION
# ================================================================

function crossValidationCNN(modelBuilder, X, y)

    n = length(y)

    indices = collect(1:n)

    Random.shuffle!(indices)

    folds = MLUtils.kfolds(indices, k=K_FOLDS)

    metricsList = []

    globalCM = zeros(Float64,
        numClasses,
        numClasses)

    for (fold, (train_idx, test_idx)) in enumerate(folds)

        println("\nFold $fold / $K_FOLDS")

        Xtrain = X[:,:,:,train_idx]
        Xtest  = X[:,:,:,test_idx]

        yTrain = y[train_idx]
        yTest  = y[test_idx]

        yTrainOH = onehotbatch(yTrain, classes)

        model = modelBuilder()

        trainCNN(model, Xtrain, yTrainOH)

        preds = model(Xtest)

        predLabels = onecold(preds, classes)

        cm = confusionMatrix(
            yTest,
            predLabels,
            classes
        )

        globalCM .+= cm

        metrics = metricsFromConfusionMatrix(cm)

        push!(metricsList, metrics)

        println(
            "Accuracy fold: ",
            round(metrics.accuracy, digits=4)
        )

        println(
            "F1 fold: ",
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

        "cm" => globalCM ./ K_FOLDS

    )

    return result

end

# ================================================================
# SAVE REPORT
# ================================================================

function saveReport(results)

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
    println(io, "Semilla aleatoria: 1234")
    println(io, "Validación cruzada: $K_FOLDS folds")

    println(io)

    modelos = join(keys(results), ", ")

    println(io, "Arquitecturas ejecutadas: $modelos")

    println(io)
    println(io,
"============================================================")
    println(io)

    bestF1 = -1.0
    bestName = ""

    for (name,res) in results

        if res["f1_mean"] > bestF1

            bestF1 = res["f1_mean"]
            bestName = name

        end
    end

    for (name,res) in results

        println(io,
"╔════════════════════════════════════════════════════════════╗")

        println(io,
"║  ARQUITECTURA: $(uppercase(name))")

        println(io,
"╚════════════════════════════════════════════════════════════╝")

        println(io)

        println(io,
"┌─ RESULTADOS DETALLADOS ──────────────────────────────────┐")

        println(io)

        @printf(io,
            "Accuracy:      %.4f ± %.4f\n",
            res["accuracy_mean"],
            res["accuracy_std"])

        @printf(io,
            "Sensibilidad:  %.4f ± %.4f\n",
            res["sens_mean"],
            res["sens_std"])

        @printf(io,
            "Especificidad: %.4f ± %.4f\n",
            res["spec_mean"],
            res["spec_std"])

        @printf(io,
            "Precision:     %.4f ± %.4f\n",
            res["prec_mean"],
            res["prec_std"])

        @printf(io,
            "F1-Score:      %.4f ± %.4f\n",
            res["f1_mean"],
            res["f1_std"])

        println(io)

        println(io,
"└──────────────────────────────────────────────────────────┘")

        println(io)

        println(io,
"┌─ MATRIZ DE CONFUSIÓN ────────────────────────────────────┐")

        cm = res["cm"]

        header = @sprintf("%14s", "")

        for c in classes

            cname = uppercase(first(c, min(5,length(c))))

            header *= @sprintf("%12s", "Pred:$cname")

        end

        println(io, header)

        for i in 1:length(classes)

            row = @sprintf("%12s",
                "Real:" * classes[i])

            for j in 1:length(classes)

                row *= @sprintf(
                    "%12.1f",
                    cm[i,j]
                )

            end

            println(io, row)

        end

        println(io,
"└──────────────────────────────────────────────────────────┘")

        println(io)
        println(io,
"------------------------------------------------------------")
        println(io)

    end

    println(io,
"════════════════════════════════════════════════════════════")

    println(io,
        "MEJOR ARQUITECTURA: $(uppercase(bestName))")

    @printf(io,
        "F1-score: %.4f\n",
        bestF1)

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

println("\n================================================")
println("INICIO DE ENTRENAMIENTO CNN")
println("================================================\n")

results = Dict()

architectures = [

    ("CNN_1", () -> buildCnn1(numClasses)),
    ("CNN_2", () -> buildCnn2(numClasses)),
    ("CNN_3", () -> buildCnn3(numClasses)),
    ("CNN_4", () -> buildCnn4(numClasses)),
    ("CNN_5", () -> buildCnn5(numClasses)),
    ("CNN_6", () -> buildCnn6(numClasses)),
    ("CNN_7", () -> buildCnn7(numClasses)),
    ("CNN_8", () -> buildCnn8(numClasses)),
    ("CNN_9", () -> buildCnn9(numClasses)),
    ("CNN_10", () -> buildCnn10(numClasses))

]

for (name, builder) in architectures

    println("\n========================================")
    println("Entrenando arquitectura: $name")
    println("========================================\n")

    result = crossValidationCNN(
        builder,
        X,
        y
    )

    results[name] = result

    println(
        "\nF1-score: ",
        round(result["f1_mean"], digits=4)
    )

end

saveReport(results)

println("\nFIN.")