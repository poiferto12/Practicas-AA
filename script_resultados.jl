# script_resultados.jl
# Script maestro modular para ejecutar todos los modelos y generar reporte
# 
# Características:
# - Modular: ejecuta solo lo que esté disponible
# - Exporta resultados a archivo de texto formateado
# - Genera resúmenes por modelo y comparativas

using Random
using Dates
Random.seed!(1234)

include("firmas.jl");
include("extractFeatures.jl");

nothing  # Suprimir salida de includes

# ============================================================================
# FUNCIÓN AUXILIAR: Formatear matrices de confusión
# ============================================================================

function formatConfusionMatrix(cm::AbstractMatrix)
    """Formatea una matriz de confusión para mostrarla de forma legible"""
    classes = ["cats", "dogs", "cows", "frogs"]
    n_classes = size(cm, 1)
    
    lines = String[]
    
    # Encabezado
    header = "     " 
    for j in 1:n_classes
        header *= rpad(" Pred: $(uppercase(classes[j][1:min(3, length(classes[j]))]))", 15)
    end
    push!(lines, header)
    
    # Filas
    for i in 1:n_classes
        line = "Real: $(rpad(classes[i], 4))"
        for j in 1:n_classes
            val = round(cm[i, j], digits=1)
            line *= rpad(" $val", 15)
        end
        push!(lines, line)
    end
    
    return join(lines, "\n")
end

# ============================================================================
# SECCIÓN 1: CARGAR DATOS (UNA SOLA VEZ)
# ============================================================================

println("┌─────────────────────────────────────────┐")
println("│  CARGANDO DATASET                       │")
println("└─────────────────────────────────────────┘")
println()

try
    inputs, targets = loadDataset("./dataset")
    cvIndices = crossvalidation(targets, 10)
    println("✓ Dataset cargado correctamente")
    println("  - Muestras: $(size(inputs, 1))")
    println("  - Características: $(size(inputs, 2))")
    println("  - Clases: $(length(unique(targets)))")
    println("  - Folds de validación cruzada: 10")
    println()
catch e
    println("✗ Error cargando dataset: ", e)
    exit(1)
end

# ============================================================================
# SECCIÓN 2: DICCIONARIO PARA ALMACENAR TODOS LOS RESULTADOS
# ============================================================================

resultados_globales = Dict{String, Any}()
modelos_ejecutados = String[]

# ============================================================================
# SECCIÓN 3: EJECUTAR REDES NEURONALES
# ============================================================================

println("┌─────────────────────────────────────────┐")
println("│  EJECUTANDO: Redes Neuronales          │")
println("└─────────────────────────────────────────┘")
println()

try
    if isfile("aproxANN.jl")
        include("aproxANN.jl")
        
        # Verificar si se ejecutó bien capturando la excepción
        try
            if !isempty(results)
                push!(modelos_ejecutados, "ANN")
                resultados_globales["ANN"] = (results, best_idx, best_conf_matrix, train_accuracy, train_recall, train_specificity, train_precision, train_f1, train_conf_matrix)
                println("\n✓ Redes Neuronales ejecutadas correctamente\n")
            else
                println("✗ No se generaron resultados en aproxANN.jl\n")
            end
        catch err
            if isa(err, UndefVarError)
                println("✗ No se generaron resultados en aproxANN.jl\n")
            else
                throw(err)
            end
        end
    else
        println("✗ Archivo no encontrado: aproxANN.jl\n")
    end
catch e
    println("✗ Error en Redes Neuronales: ")
    println("  $(typeof(e)): $(e)\n")
end

# ============================================================================
# SECCIÓN 4: EJECUTAR SVM
# ============================================================================

println("┌─────────────────────────────────────────┐")
println("│  EJECUTANDO: SVM                        │")
println("└─────────────────────────────────────────┘")
println()

try
    if isfile("aproxSVM.jl")
        include("aproxSVM.jl")
        
        # Verificar si se ejecutó bien
        try
            if !isempty(results)
                push!(modelos_ejecutados, "SVM")
                resultados_globales["SVM"] = (results, best_config, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix)
                println("\n✓ SVM ejecutado correctamente\n")
            else
                println("✗ No se generaron resultados en aproxSVM.jl\n")
            end
        catch err
            if isa(err, UndefVarError)
                println("✗ No se generaron resultados en aproxSVM.jl\n")
            else
                throw(err)
            end
        end
    else
        println("✗ Archivo no encontrado: aproxSVM.jl\n")
    end
catch e
    println("✗ Error en SVM: ")
    println("  $(typeof(e)): $(e)\n")
end

# ============================================================================
# SECCIÓN 5: EJECUTAR kNN (si existe)
# ============================================================================

println("┌─────────────────────────────────────────┐")
println("│  EJECUTANDO: kNN                        │")
println("└─────────────────────────────────────────┘")
println()

try
    if isfile("aproxKNN.jl")
        include("aproxKNN.jl")
        
        # Verificar si se ejecutó bien
        try
            if !isempty(results)
                push!(modelos_ejecutados, "kNN")
                resultados_globales["kNN"] = (results, best_k, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix)
                println("\n✓ kNN ejecutado correctamente\n")
            else
                println("✗ No se generaron resultados en aproxKNN.jl\n")
            end
        catch err
            if isa(err, UndefVarError)
                println("✗ No se generaron resultados en aproxKNN.jl\n")
            else
                throw(err)
            end
        end
    else
        println("✗ Archivo no encontrado: aproxKNN.jl")
        println("  (Continuando con los demás modelos...)\n")
    end
catch e
    println("✗ Error en kNN: ")
    println("  $(typeof(e)): $(e)")
    println("  (Continuando con los demás modelos...)\n")
end

# ============================================================================
# SECCIÓN 6: EJECUTAR DECISION TREE
# ============================================================================

println("┌─────────────────────────────────────────┐")
println("│  EJECUTANDO: Decision Tree              │")
println("└─────────────────────────────────────────┘")
println()

try
    if isfile("aproxDecisionTree.jl")
        include("aproxDecisionTree.jl")
        
        try
            if !isempty(results)
                push!(modelos_ejecutados, "DecisionTree")
                resultados_globales["DecisionTree"] = (results, best_depth, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix)
                println("\n✓ Árboles de Decisión ejecutados correctamente\n")
            else
                println("✗ No se generaron resultados en aproxDecisionTree.jl\n")
            end
        catch err
            if isa(err, UndefVarError)
                println("✗ No se generaron resultados en aproxDecisionTree.jl\n")
            else
                throw(err)
            end
        end
    else
        println("✗ Archivo no encontrado: aproxDecisionTree.jl\n")
    end
catch e
    println("✗ Error en Decision Tree: ")
    println("  $(typeof(e)): $(e)\n")
end

# ============================================================================
# SECCIÓN 7: EJECUTAR DoME
# ============================================================================

println("┌─────────────────────────────────────────┐")
println("│  EJECUTANDO: DoME                       │")
println("└─────────────────────────────────────────┘")
println()

try
    if isfile("aproxDoME.jl")
        include("aproxDoME.jl")
        
        try
            if !isempty(results)
                push!(modelos_ejecutados, "DoME")
                resultados_globales["DoME"] = (results, best_nodes, best_conf_matrix, train_accuracy, train_f1, train_conf_matrix)
                println("\n✓ DoME ejecutado correctamente\n")
            else
                println("✗ No se generaron resultados en aproxDoME.jl\n")
            end
        catch err
            if isa(err, UndefVarError)
                println("✗ No se generaron resultados en aproxDoME.jl\n")
            else
                throw(err)
            end
        end
    else
        println("✗ Archivo no encontrado: aproxDoME.jl\n")
    end
catch e
    println("✗ Error en DoME: ")
    println("  $(typeof(e)): $(e)\n")
end

# ============================================================================
# SECCIÓN 8: GENERAR REPORTE EN ARCHIVO .TXT
# ============================================================================

println()
println("="^60)
println("GENERANDO REPORTE DE RESULTADOS")
println("="^60)
println()

if isempty(modelos_ejecutados)
    println("✗ No se ejecutó ningún modelo correctamente.")
    println("  Revisa los errores anteriores.")
else
    # Abrir archivo para escritura
    open("REPORTE_RESULTADOS.txt", "w") do io
        
        write(io, "╔════════════════════════════════════════════════════════════╗\n")
        write(io, "║  REPORTE DE RESULTADOS - CLASIFICACIÓN DE SONIDOS ANIMALES  ║\n")
        write(io, "╚════════════════════════════════════════════════════════════╝\n\n")
        
        write(io, "Fecha de ejecución: $(Dates.now())\n")
        write(io, "Semilla aleatoria: 1234\n")
        write(io, "Validación cruzada: 10 folds\n\n")
        
        write(io, "Modelos ejecutados: $(join(modelos_ejecutados, ", "))\n")
        write(io, "Modelos faltantes: ")
        modelos_faltantes = setdiff(["ANN", "SVM", "kNN"], modelos_ejecutados)
        if isempty(modelos_faltantes)
            write(io, "Ninguno\n")
        else
            write(io, "$(join(modelos_faltantes, ", "))\n")
        end
        write(io, "\n" * "="^60 * "\n\n")
    
    # ========================================================================
    # SECCIÓN ANN
    # ========================================================================
    
    if in("ANN", modelos_ejecutados)
        (ann_results, ann_best_idx, ann_conf_matrix, ann_train_accuracy, ann_train_recall, ann_train_specificity, ann_train_precision, ann_train_f1, ann_train_conf_matrix) = resultados_globales["ANN"]
        
        write(io, "\n╔════════════════════════════════════════════════════════════╗\n")
        write(io, "║  MODELO: REDES NEURONALES ARTIFICIALES (ANN)\n")
        write(io, "╚════════════════════════════════════════════════════════════╝\n\n")
        
        write(io, "TOPOLOGÍAS PROBADAS: $(length(ann_results))\n\n")
        
        write(io, "┌─ RESULTADOS DETALLADOS ─────────────────────────────────┐\n")
        for (idx, resultado) in enumerate(ann_results)
            topology = resultado.topology
            write(io, "\n$(idx). Topología: $topology\n")
            write(io, "   Accuracy:      $(round(resultado.accuracy[1], digits=4)) ± $(round(resultado.accuracy[2], digits=4))\n")
            write(io, "   Sensibilidad:  $(round(resultado.recall[1], digits=4)) ± $(round(resultado.recall[2], digits=4))\n")
            write(io, "   Especificidad: $(round(resultado.specificity[1], digits=4)) ± $(round(resultado.specificity[2], digits=4))\n")
            write(io, "   VPP (Precision): $(round(resultado.precision[1], digits=4)) ± $(round(resultado.precision[2], digits=4))\n")
            write(io, "   F1-Score:      $(round(resultado.f1[1], digits=4)) ± $(round(resultado.f1[2], digits=4))\n")
        end
        write(io, "\n└──────────────────────────────────────────────────────────┘\n")
        
        best_ann = ann_results[ann_best_idx]
        write(io, "\n┌─ MEJOR TOPOLOGÍA (por F1-score) ────────────────────────┐\n")
        write(io, "│ Configuración: $(best_ann.topology)\n")
        write(io, "│ Accuracy:      $(round(best_ann.accuracy[1], digits=4)) ± $(round(best_ann.accuracy[2], digits=4))\n")
        write(io, "│ Sensibilidad:  $(round(best_ann.recall[1], digits=4)) ± $(round(best_ann.recall[2], digits=4))\n")
        write(io, "│ Especificidad: $(round(best_ann.specificity[1], digits=4)) ± $(round(best_ann.specificity[2], digits=4))\n")
        write(io, "│ VPP:           $(round(best_ann.precision[1], digits=4)) ± $(round(best_ann.precision[2], digits=4))\n")
        write(io, "│ F1-Score:      $(round(best_ann.f1[1], digits=4)) ± $(round(best_ann.f1[2], digits=4))\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ DE CONFUSIÓN (promedio 10-fold) ────────────────┐\n")
        if typeof(ann_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(ann_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MÉTRICAS DE TRAINING (Todo el dataset) ─────────────────┐\n")
        write(io, "│ Accuracy:      $(round(ann_train_accuracy[1], digits=4))\n")
        write(io, "│ Sensibilidad:  $(round(ann_train_recall[1], digits=4))\n")
        write(io, "│ Especificidad: $(round(ann_train_specificity[1], digits=4))\n")
        write(io, "│ VPP:           $(round(ann_train_precision[1], digits=4))\n")
        write(io, "│ F1-Score:      $(round(ann_train_f1[1], digits=4))\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ CONFUSIÓN TRAINING ───────────────────────────────┐\n")
        if typeof(ann_train_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(ann_train_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        write(io, "\n" * "-"^60 * "\n")
    end
    
    # ========================================================================
    # SECCIÓN SVM
    # ========================================================================
    
    if in("SVM", modelos_ejecutados)
        (svm_results, svm_best_config, svm_conf_matrix, svm_train_accuracy, svm_train_f1, svm_train_conf_matrix) = resultados_globales["SVM"]
        
        write(io, "\n╔════════════════════════════════════════════════════════════╗\n")
        write(io, "║  MODELO: MÁQUINAS DE VECTORES DE SOPORTE (SVM)\n")
        write(io, "╚════════════════════════════════════════════════════════════╝\n\n")
        
        write(io, "CONFIGURACIONES PROBADAS: $(length(svm_results))\n\n")
        
        write(io, "┌─ RESULTADOS DETALLADOS ─────────────────────────────────┐\n")
        for (idx, (config, res)) in enumerate(svm_results)
            write(io, "\n$(idx). Kernel: $(config["kernel"]), C: $(config["C"])\n")
            write(io, "   Accuracy:      $(round(res[1][1], digits=4)) ± $(round(res[1][2], digits=4))\n")
            write(io, "   Sensibilidad:  $(round(res[3][1], digits=4)) ± $(round(res[3][2], digits=4))\n")
            write(io, "   Especificidad: $(round(res[4][1], digits=4)) ± $(round(res[4][2], digits=4))\n")
            write(io, "   VPP (Precision): $(round(res[5][1], digits=4)) ± $(round(res[5][2], digits=4))\n")
            write(io, "   F1-Score:      $(round(res[7][1], digits=4)) ± $(round(res[7][2], digits=4))\n")
        end
        write(io, "\n└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MEJOR CONFIGURACIÓN (por F1-score) ──────────────────────┐\n")
        write(io, "│ Kernel: $(svm_best_config["kernel"])\n")
        write(io, "│ C:      $(svm_best_config["C"])\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ DE CONFUSIÓN (promedio 10-fold) ──────────────────┐\n")
        if typeof(svm_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(svm_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MÉTRICAS DE TRAINING (Todo el dataset) ─────────────────┐\n")
        write(io, "│ Accuracy:      $(round(svm_train_accuracy[1], digits=4))\n")
        write(io, "│ F1-Score:      $(round(svm_train_f1[1], digits=4))\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ CONFUSIÓN TRAINING ───────────────────────────────┐\n")
        if typeof(svm_train_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(svm_train_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        write(io, "\n" * "-"^60 * "\n")
    end
    
    # ========================================================================
    # SECCIÓN kNN (si existe)
    # ========================================================================
    
    if in("kNN", modelos_ejecutados)
        (knn_results, knn_best_k, knn_conf_matrix, knn_train_accuracy, knn_train_f1, knn_train_conf_matrix) = resultados_globales["kNN"]
        
        write(io, "\n╔════════════════════════════════════════════════════════════╗\n")
        write(io, "║  MODELO: k-NEAREST NEIGHBORS (kNN)\n")
        write(io, "╚════════════════════════════════════════════════════════════╝\n\n")
        
        write(io, "VALORES DE k PROBADOS: $(length(knn_results))\n\n")
        
        write(io, "┌─ RESULTADOS DETALLADOS ─────────────────────────────────┐\n")
        for k in sort(collect(keys(knn_results)))
            res = knn_results[k]
            write(io, "\nk=$k:\n")
            write(io, "   Accuracy:      $(round(res[1][1], digits=4)) ± $(round(res[1][2], digits=4))\n")
            write(io, "   Sensibilidad:  $(round(res[3][1], digits=4)) ± $(round(res[3][2], digits=4))\n")
            write(io, "   Especificidad: $(round(res[4][1], digits=4)) ± $(round(res[4][2], digits=4))\n")
            write(io, "   VPP (Precision): $(round(res[5][1], digits=4)) ± $(round(res[5][2], digits=4))\n")
            write(io, "   F1-Score:      $(round(res[7][1], digits=4)) ± $(round(res[7][2], digits=4))\n")
        end
        write(io, "\n└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MEJOR k (por F1-score) ───────────────────────────────────┐\n")
        write(io, "│ k = $knn_best_k\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ DE CONFUSIÓN (promedio 10-fold) ──────────────────┐\n")
        if typeof(knn_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(knn_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MÉTRICAS DE TRAINING (Todo el dataset) ─────────────────┐\n")
        write(io, "│ Accuracy:      $(round(knn_train_accuracy[1], digits=4))\n")
        write(io, "│ F1-Score:      $(round(knn_train_f1[1], digits=4))\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ CONFUSIÓN TRAINING ───────────────────────────────┐\n")
        if typeof(knn_train_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(knn_train_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        write(io, "\n" * "-"^60 * "\n")
    end

    # ========================================================================
    # SECCIÓN DECISION TREE
    # ========================================================================

    if in("DecisionTree", modelos_ejecutados)
        (dt_results, dt_best_depth, dt_conf_matrix, dt_train_accuracy, dt_train_f1, dt_train_conf_matrix) = resultados_globales["DecisionTree"]
    
        write(io, "\n╔════════════════════════════════════════════════════════════╗\n")
        write(io, "║  MODELO: ÁRBOLES DE DECISIÓN\n")
        write(io, "╚════════════════════════════════════════════════════════════╝\n\n")
    
        write(io, "PROFUNDIDADES PROBADAS: $(length(dt_results))\n\n")
        
        write(io, "┌─ RESULTADOS DETALLADOS ─────────────────────────────────┐\n")
        for depth in sort(collect(keys(dt_results)))
            res = dt_results[depth]
            write(io, "\nProfundidad=$depth:\n")
            write(io, "   Accuracy:      $(round(res[1][1], digits=4)) ± $(round(res[1][2], digits=4))\n")
            write(io, "   Sensibilidad:  $(round(res[3][1], digits=4)) ± $(round(res[3][2], digits=4))\n")
            write(io, "   Especificidad: $(round(res[4][1], digits=4)) ± $(round(res[4][2], digits=4))\n")
            write(io, "   VPP (Precision): $(round(res[5][1], digits=4)) ± $(round(res[5][2], digits=4))\n")
            write(io, "   F1-Score:      $(round(res[7][1], digits=4)) ± $(round(res[7][2], digits=4))\n")
        end
        write(io, "\n└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MEJOR PROFUNDIDAD (por F1-score) ────────────────────────┐\n")
        write(io, "│ max_depth = $dt_best_depth\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
    
        write(io, "\n┌─ MATRIZ DE CONFUSIÓN (promedio 10-fold) ──────────────────┐\n")
        if typeof(dt_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(dt_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MÉTRICAS DE TRAINING (Todo el dataset) ─────────────────┐\n")
        write(io, "│ Accuracy:      $(round(dt_train_accuracy[1], digits=4))\n")
        write(io, "│ F1-Score:      $(round(dt_train_f1[1], digits=4))\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ CONFUSIÓN TRAINING ───────────────────────────────┐\n")
        if typeof(dt_train_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(dt_train_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        write(io, "\n" * "-"^60 * "\n")
    end

    # ========================================================================
    # SECCIÓN DoME
    # ========================================================================

    if in("DoME", modelos_ejecutados)
        (dome_results, dome_best_nodes, dome_conf_matrix, dome_train_accuracy, dome_train_f1, dome_train_conf_matrix) = resultados_globales["DoME"]
    
        write(io, "\n╔════════════════════════════════════════════════════════════╗\n")
        write(io, "║  MODELO: DoME\n")
        write(io, "╚════════════════════════════════════════════════════════════╝\n\n")
    
        write(io, "NODOS PROBADOS: $(length(dome_results))\n\n")
        
        write(io, "┌─ RESULTADOS DETALLADOS ─────────────────────────────────┐\n")
        for n in sort(collect(keys(dome_results)))
            res = dome_results[n]
            write(io, "\nNodos=$n:\n")
            write(io, "   Accuracy:      $(round(res[1][1], digits=4)) ± $(round(res[1][2], digits=4))\n")
            write(io, "   Sensibilidad:  $(round(res[3][1], digits=4)) ± $(round(res[3][2], digits=4))\n")
            write(io, "   Especificidad: $(round(res[4][1], digits=4)) ± $(round(res[4][2], digits=4))\n")
            write(io, "   VPP (Precision): $(round(res[5][1], digits=4)) ± $(round(res[5][2], digits=4))\n")
            write(io, "   F1-Score:      $(round(res[7][1], digits=4)) ± $(round(res[7][2], digits=4))\n")
        end
        write(io, "\n└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MEJOR CONFIGURACIÓN (por F1-score) ──────────────────────┐\n")
        write(io, "│ maximumNodes = $dome_best_nodes\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
    
        write(io, "\n┌─ MATRIZ DE CONFUSIÓN (promedio 10-fold) ──────────────────┐\n")
        if typeof(dome_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(dome_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MÉTRICAS DE TRAINING (Todo el dataset) ─────────────────┐\n")
        write(io, "│ Accuracy:      $(round(dome_train_accuracy[1], digits=4))\n")
        write(io, "│ F1-Score:      $(round(dome_train_f1[1], digits=4))\n")
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        
        write(io, "\n┌─ MATRIZ CONFUSIÓN TRAINING ───────────────────────────────┐\n")
        if typeof(dome_train_conf_matrix) <: AbstractMatrix
            write(io, formatConfusionMatrix(dome_train_conf_matrix) * "\n")
        end
        write(io, "└──────────────────────────────────────────────────────────┘\n")
        write(io, "\n" * "-"^60 * "\n")
    end
    
    write(io, "\n")
    write(io, "═"^60 * "\n")
    write(io, "FIN DEL REPORTE\n")
    write(io, "═"^60 * "\n")
    end
    
    println("✓ Reporte guardado en: REPORTE_RESULTADOS.txt")
    println()
    println("Modelos ejecutados: ", join(modelos_ejecutados, ", "))
    println("Archivo listo para ser procesado en la memoria LaTeX")
end

